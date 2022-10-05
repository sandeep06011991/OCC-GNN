# Reaches around 0.7866 ± 0.0041 test accuracy.

import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
import torch.multiprocessing as mp
import quiver
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from dgl_sage import SAGE
from dgl_gat import GAT

def average(ls):
    if(len(ls) == 1):
        return ls[0]
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)



def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.module.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# Entry point

def run(rank, args,  data):
    # Unpack data
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size=4)
    
    device = rank
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    if args.data == 'gpu':
        nfeat = nfeat.to(rank)
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        g = g.formats(['csc'])
        g = g.to(device)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')], replace = True)
    world_size = 4
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 if args.sample_gpu else args.num_workers,
        persistent_workers=not args.sample_gpu)

    # Define model and optimizer
    if args.model == "GCN":
        model = SAGE(in_feats, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    else:
        assert(args.model == "GAT")
        heads = 3
        model = GAT(in_feats, args.num_hidden, \
                n_classes , heads, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_val_acc = final_test_acc = 0
    print("start training")

    epoch_time = []
    sample_time_epoch = []
    movement_time_graph_epoch = []
    movement_time_feature_epoch = []
    forward_time_epoch = []
    backward_time_epoch = []
    accuracy = []
    
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    e4 = torch.cuda.Event(enable_timing = True)

    for epoch in range(args.num_epochs):
        tic = time.time()

        model.train()
        #pbar = tqdm(total=train_nid.size(0))
        #pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_correct = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_start = time.time()
        sample_time = 0
        movement_graph_time = 0
        movement_feature_time = 0
        forward_time = 0
        backward_time = 0
        dataloader_i = iter(dataloader)
        try:
            while True:
                optimizer.zero_grad()
                t1 = time.time()
                input_nodes, seeds, blocks = next(dataloader_i) 
                t2 = time.time()
                # copy block to gpu
                blocks = [blk.to(device) for blk in blocks]
                blocks = [blk.formats(['coo','csr','csc']) for blk in blocks]
                for blk in blocks:
                    blk.create_formats_()
                t3 = time.time()
                e1.record()
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(
                    nfeat, labels, seeds, input_nodes, device)
                e2.record()
                t4 = time.time()
                # Compute loss and prediction
                with torch.autograd.profiler.profile(enabled=(False), use_cuda=True, profile_memory = True) as prof:
                    batch_pred = model(blocks, batch_inputs)
                #e3.record()
                    loss = loss_fcn(batch_pred, batch_labels)
                    e3.record()
                    loss.backward()
                    e4.record()
                e4.synchronize()
                optimizer.step()
                sample_time += (t2 - t1)
                movement_graph_time += (t3 - t2)
                movement_feature_time += max(t4-t3, e1.elapsed_time(e2)/1000)
                forward_time += e2.elapsed_time(e3)/1000
                backward_time += e3.elapsed_time(e4)/1000
                #print("forward time", e2.elapsed_time(e3)/1000)
                #print("backward time", e3.elapsed_time(e4)/1000)
                total_loss += loss.item()
                total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
        #        pbar.update(args.batch_size)
        except StopIteration:
            pass
        epoch_time.append(time.time()-epoch_start)
        sample_time_epoch.append(sample_time)
        movement_time_graph_epoch.append(movement_graph_time)
        movement_time_feature_epoch.append(movement_feature_time)
        forward_time_epoch.append(forward_time)
        backward_time_epoch.append(backward_time)

        #pbar.close()

        loss = total_loss / len(dataloader)
        approx_acc = total_correct / (len(dataloader) * args.batch_size)
        accuracy.append(approx_acc)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}, Epoch Time: {time.time() - tic:.4f}')
    if rank == 0:
        #print(prof.key_averages().table(sort_by='cuda_time_total'))

        if epoch >= 10:
            val_acc, test_acc = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
            print(f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    if rank == 0:
        print("accuracy:{}".format(accuracy[-1]))
        print("epoch:{}".format(average(epoch_time)))
        print("sample_time:{}".format(average(sample_time_epoch)))
        print("movement graph:{}".format(average(movement_time_graph_epoch)))
        print("movement feature:{}".format(average(movement_time_feature_epoch)))
        print("forward time:{}".format(average(forward_time_epoch)))
        print("backward time:{}".format(average(backward_time_epoch)))
    dist.destroy_process_group()

    return final_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str)
    argparser.add_argument('--cache-per', type =float)
    argparser.add_argument('--model',type = str)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--sample-gpu', action='store_true')
    argparser.add_argument('--data', type=str, choices=('cpu', 'gpu', 'quiver', 'unified'), default = 'quiver')

    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load ogbn-products data
    root = "/mnt/bigdata/sandeep/"
    data = DglNodePropPredDataset(name=args.graph, root=root)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    graph = graph.add_self_loop()
    labels = labels[:, 0].to(device)
    
    feat = graph.ndata.pop('feat')
    #year = graph.ndata.pop('year')
    if args.data == 'cpu':
        nfeat = feat
    elif args.data == 'gpu':
        nfeat = feat.to(device)
    elif args.data == 'quiver':
        quiver.init_p2p(device_list = [0,1,2,3])
        csr_topo = quiver.CSRTopo(th.stack(graph.edges('uv')))
        cache_size = int(float(args.cache_per) * feat.shape[0] * feat.shape[1] * 4/(1024 * 1024))
        device_cache_size = "{}M".format(cache_size)
        print("calculated cache_size ", cache_size)
        if float(args.cache_per) > .25:
            cache_policy = "device_replicate"
        else:
            cache_policy = "p2p_clique_replicate"
        nfeat = quiver.Feature(rank=0, device_list=[0,1,2,3], 
                               #device_cache_size="200M",
                               device_cache_size = device_cache_size, 
                               cache_policy = cache_policy) 
                               #cache_policy="device_replicate", 
                               #csr_topo=csr_topo)
        nfeat.from_cpu_tensor(feat)
    elif args.data == 'unified':
        from distutils.version import LooseVersion
        assert LooseVersion(dgl.__version__) >= LooseVersion('0.8.0'), \
            f'Current DGL version ({dgl.__version__}) does not support UnifiedTensor.'
        feat = dgl.contrib.UnifiedTensor(feat, device=device)
    else:
        raise ValueError(f'Unsupported feature storage location {args.data}.')

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph

    #test_accs = []
    #for i in range(1, 11):
    #    print(f'\nRun {i:02d}:\n')
    #    test_acc = run(args, device, data)
    #    test_accs.append(test_acc)
    #test_accs = th.tensor(test_accs)
    print('============================')
    #print(f'Final Test: {test_accs.mean():.4f} ± {test_accs.std():.4f}')
    world_size = 4
    mp.spawn(
        run,
        args=(args,  data),
        nprocs=world_size,
        join=True
    )

