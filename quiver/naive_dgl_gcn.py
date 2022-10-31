# Reaches around 0.7866 ± 0.0041 test accuracy.
import logging
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
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from dgl_sage import SAGE
from dgl_gat import GAT

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
        PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
        PATH_DIR = "/home/spolisetty/OCC-GNN"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
        PATH_DIR = "/home/q91/OCC-GNN"
    return DATA_DIR,PATH_DIR

DATA_DIR, PATH_DIR = get_data_dir()
def average(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, test_nid, device):
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
    return compute_acc(pred[test_nid], labels[test_nid].to(device))
    # model.train()
    # return compute_acc(pred[test_nid], labels[test_nid].to(device))


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    t1 = time.time()
    assert(nfeat.device == torch.device('cpu'))
    assert(labels.device == torch.device('cpu'))

    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    t2 = time.time()
    return batch_inputs, batch_labels

# Entry point

def run(rank, args,  data):
    # Unpack data
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size=4)

    device = rank
    torch.cuda.set_device(device)
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, offsets, test_graph = data
    if rank == 0:
      os.makedirs('{}/quiver/logs_naive'.format(PATH_DIR),exist_ok = True)
      FILENAME= ('{}/quiver/logs_naive/{}_{}_{}.txt'.format(PATH_DIR, \
               args.graph, args.batch_size, args.model))

      fileh = logging.FileHandler(FILENAME, 'w')
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      fileh.setFormatter(formatter)

      log = logging.getLogger()  # root logger
      log.addHandler(fileh)      # set the new handler
      log.setLevel(logging.INFO)

    #if args.data == 'gpu':
    #    nfeat = nfeat.to(rank)
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
    number_of_minibatches = train_nid.shape[0]/args.batch_size
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cpu',
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor = 6,
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
    data_movement_epoch = []
    forward_time_epoch = []
    backward_time_epoch = []
    accuracy = []
    edges_per_epoch = []
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    e4 = torch.cuda.Event(enable_timing = True)
    test_accuracy_list = []
    prof = torch.autograd.profiler.profile(enabled=(rank==0), use_cuda=True, profile_memory = True) 
    prof.__enter__()
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
        edges_computed = 0
        data_movement = 0
        global_cache_misses = 0
        step = 0
        try:
            while True:
                optimizer.zero_grad()
                t1 = time.time()
                input_nodes, seeds, blocks = next(dataloader_i)
                t2 = time.time()
                # copy block to gpu
                blocks = [blk.to(device) for blk in blocks]
                #t2 = time.time()
                blocks = [blk.formats(['coo','csr','csc']) for blk in blocks]

                for blk in blocks:
                    blk.create_formats_()
                    edges_computed += blk.edges()[0].shape[0]
                #print(edges_computed)
                t3 = time.time()
                #start = offsets[device][0]
                #end = offsets[device][1]
                #hit = torch.where((input_nodes > start) & (input_nodes < end))[0].shape[0]

                #hit = torch.where(input_nodes < offsets[3])[0].shape[0]
                missed = input_nodes.shape[0] 

                e1.record()
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(
                    nfeat, labels, seeds, input_nodes, device)
                e2.record()
                t4 = time.time()
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                print("start iteration")
                #with torch.autograd.profiler.record_function('model_loss'):
                for j in range(10):
                    e2.record()
                    batch_pred = model(blocks, batch_inputs)
                    e3.record(torch.cuda.current_stream())
                    print(torch.cuda.current_stream())
                    loss = loss_fcn(batch_pred, batch_labels)
                    #e3.record()
                    loss.backward()
                    e4.record(torch.cuda.current_stream())
                    e4.synchronize()
                    print("forward time", e2.elapsed_time(e3)/1000)
                    print("backward time",e3.elapsed_time(e4)/1000)
                print("end iteration")
                optimizer.step()
                sample_time += (t2 - t1)
                movement_graph_time += (t3 - t2)
                #print("sample time", t2-t1, t3-t2)
                #print("Time feature", device, e1.elapsed_time(e2)/1000)
                #print("Expected bandwidth", missed * nfeat.shape[1] * 4/ ((e1.elapsed_time(e2)/1000) * 1024 * 1024 * 1024), "GB device", rank, "cache rate", hit/(hit + missed))
                movement_feature_time += max(t4-t3, e1.elapsed_time(e2)/1000)
                forward_time += e2.elapsed_time(e3)/1000
                backward_time += e3.elapsed_time(e4)/1000
                data_movement += (batch_inputs.shape[0] * nfeat.shape[1] * 4/(1024 * 1024))
                if args.early_stopping and step ==20:
                    break
                step = step + 1
                if rank == 0:
                    log.info("step {}, epoch {}, number_of_minibatches {}".format(step, epoch, number_of_minibatches))

                print("forward time", forward_time)
                print("backward time", backward_time)
                total_loss += loss.item()
                total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
        #        pbar.update(args.batch_size)
        except StopIteration:
            pass
        if test_graph != None and rank== 0:
            test_nid = torch.where(test_graph.ndata['test_mask'])[0]
            test_accuracy = evaluate(model, test_graph, test_graph.ndata['features'], test_graph.ndata['labels'], \
                        test_nid, device)
            test_accuracy_list.append(test_accuracy.item())
            print("Accuracy: {}, device:{}, epoch:{}".format(test_accuracy, device, epoch))

        if rank == 0:
            log.info("log_accuracy:{}".format(accuracy))
            log.info("log_epoch:{}".format(epoch_time))
            log.info("log_sample_time:{}".format(sample_time_epoch))
            log.info("log_movement graph:{}".format(movement_time_graph_epoch))
            log.info("log_movement feature:{}".format(movement_time_feature_epoch))
            log.info("log_forward time:{}".format(forward_time_epoch))
            log.info("log_backward time:{}".format(backward_time_epoch))
            log.info("log_edges per epoch:{}".format(edges_per_epoch))
            log.info(data_movement_epoch)
            log.info("log_data movement:{}MB".format(data_movement_epoch))
        print("EPOCH TIME",time.time() - epoch_start)
        epoch_time.append(time.time()-epoch_start)
        sample_time_epoch.append(sample_time)
        movement_time_graph_epoch.append(movement_graph_time)
        movement_time_feature_epoch.append(movement_feature_time)
        forward_time_epoch.append(forward_time)
        backward_time_epoch.append(backward_time)
        edges_per_epoch.append(edges_computed)
        data_movement_epoch.append(data_movement)
        #pbar.close()

        loss = total_loss / len(dataloader)
        approx_acc = total_correct / (len(dataloader) * args.batch_size)
        accuracy.append(approx_acc)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}, Epoch Time: {time.time() - tic:.4f}')
    if rank == 0 :
        prof.__exit__()
        print(prof.key_averages().table(sort_by='cuda_time_total'))
        print("Test Accuracy ###########", test_accuracy_list)
        #
        # if epoch >= 10:
        #     val_acc, test_acc = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
        #     print(f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        #
        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         final_test_acc = test_acc
    print("edges per epoch:{}".format(average(edges_per_epoch)))
    if rank == 3:
        print("accuracy:{}".format(accuracy[-1]))
        print("epoch_time:{}".format(average(epoch_time)))
        print("sample_time:{}".format(average(sample_time_epoch)))
        print("movement graph:{}".format(average(movement_time_graph_epoch)))
        print("movement feature:{}".format(average(movement_time_feature_epoch)))
        print("forward time:{}".format(average(forward_time_epoch)))
        print("backward time:{}".format(average(backward_time_epoch)))
        print("edges per epoch:{}".format(average(edges_per_epoch)))
        print(data_movement_epoch)
        print("data movement:{}MB".format(average(data_movement_epoch)))
    #torch.distributed.barrier()
    #dist.destroy_process_group()
    return final_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default = 0)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--sample-gpu', action='store_true')
    argparser.add_argument('--early-stopping', action = 'store_true')
    argparser.add_argument('--test-graph',type = str)
    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load ogbn-products data
    #root = "/mnt/bigdata/sandeep/"
    import utils
    dg_graph, partition_map, num_classes = utils.get_process_graph(args.graph)
    data = dg_graph
    train_idx = torch.where(data.ndata.pop('train_mask'))[0]
    val_idx = torch.where(data.ndata.pop('val_mask'))[0]
    test_idx = torch.where(data.ndata.pop('test_mask'))[0]
    graph = dg_graph
    labels = dg_graph.ndata.pop('labels')
    nfeat = dg_graph.ndata.pop('features')
    graph = graph.add_self_loop()
    test_graph = None
    if (args.test_graph) != None:
        test_graph, _, num_classes = utils.get_process_graph(args.test_graph, True)
        test_graph = test_graph.add_self_loop()
    offsets = {3:0}
    ###################################
    #data = DglNodePropPredDataset(name=args.graph, root=root)
    #splitted_idx = data.get_idx_split()
    #train_idx, val_idx, test_idx = data.ndata['train_idx'], splitted_idx['valid'], splitted_idx['test']
    #graph, labels = data[0]
    #graph = graph.add_self_loop()
    #labels = labels[:, 0].to(device)

    # feat = graph.ndata.pop('feat')
    #year = graph.ndata.pop('year')

    in_feats = nfeat.shape[1]
    nfeat = nfeat.pin_memory()
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, offsets, test_graph

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
