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
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from dgl_sage import SAGE
from dgl_gat import GAT
from utils.async_timing_analysis import *
from utils.data.env import *
from utils.utils import *
import pickle
import nvtx
import gc
def average(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)

#ROOT_DIR ="/home/q91/torch-quiver/srcs/python"
'''import sys
def check_path():
    path_set = False
    for p in sys.path:
        if ROOT_DIR in p:
            path_set = True
    if (not path_set):
        print(sys.path)
        sys.path.append(ROOT_DIR)
        print("Setting Path")
        print(sys.path)

check_path()
'''
import quiver
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
    batch_inputs = nfeat[input_nodes.to(torch.int64)].to(device)
    batch_labels = labels[seeds.to(torch.int64)].to(device)
    return batch_inputs, batch_labels

# Entry point

def run(rank, args,  data):
    # Unpack data
    torch.cuda.set_device(rank)
    #print(feature.cpu_part)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size=4)
    device = rank
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, offsets, metrics_queue, feat_dtype= data
    nfeat.lazy_init_from_ipc_handle()
    #print("Feature order", nfeat.feature_order.shape)
    #print("Max", torch.max(nfeat.feature_order), offsets)
    #print("Total nodes", g.num_nodes())
    if args.data == 'gpu':
        nfeat = nfeat.to(rank)
    
        #g = g.to(device)

    labels = labels.to(device)
    labels[torch.where(labels == -1)[0]] = 0
    print("Corrupting labels")
    print("Cross check edges moved !")
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')], replace = False)
    world_size = 4
    print(sampler, "fanout")
    print(args.fan_out)

    print(train_nid.dtype)
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
    print("nubmer of batches", train_nid.shape[0]/args.batch_size)
    train_nid = train_nid.to(device)
    
    g = g.formats(['csc'])
        
    if args.sample_gpu:
        # copy only the csc to the GPU
        dataloader = dgl.dataloading.DataLoader(
            g.to(device),
            train_nid,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers= 0,
            persistent_workers=False)
    else:
        print("Use UVA True")
        dataloader = dgl.dataloading.DataLoader(
            g,
            train_nid,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers= 0,
            persistent_workers=False, use_uva = True)
    # Define model and optimizer
    if args.model == "GCN":
        model = SAGE(in_feats, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    else:
        assert(args.model == "GAT")
        heads = 4
        model = GAT(in_feats, args.num_hidden, \
                n_classes , heads, args.num_layers, F.relu, args.dropout)
    model = model.to(feat_dtype).to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_val_acc = final_test_acc = 0
    print("start training")

    epoch_time = []
    data_movement_cpu_epoch = []
    data_movement_gpu_epoch = []
    sample_time_epoch = []
    data_fetch_epoch = []
    forward_time_epoch = []
    backward_time_epoch = []
    epoch_metrics = []
    accuracy = []
    edges_per_epoch = []
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    e4 = torch.cuda.Event(enable_timing = True)
    e5 = torch.cuda.Event(enable_timing = True)
    epoch_available_memory = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        model.train()
        #pbar = tqdm(total=train_nid.size(0))
        #pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_correct = 0
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        epoch_start = time.time()
        dataloader_i = iter(dataloader)
        sample_time = 0
        data_fetch = 0
        forward_time = 0
        backward_time = 0
        edges_computed = 0
        data_movement_cpu = 0
        data_movement_gpu = 0
        global_cache_misses = 0
        step = 0
        minibatch_metrics = []
        try:
            while True:
                if step == 5:
                    torch.cuda.profiler.start()
                if step == 10:
                    torch.cuda.profiler.stop()

                torch.cuda.nvtx.range_push("minibatch")
                batch_time = {}
                optimizer.zero_grad()

                t1 = time.time()
                e0.record()
                input_nodes, seeds, blocks = next(dataloader_i)
                print(blocks)
                torch.cuda.nvtx.range_pop()
                e1.record()
                e1.synchronize()
                t2 = time.time()
                sample_time += t2-t1
                batch_time[SAMPLE_START_TIME] = t1
                batch_time[GRAPH_LOAD_START_TIME] = t2
                # copy block to gpu
                if not args.sample_gpu:
                    blocks = [blk.to(device) for blk in blocks]
                #t2 = time.time()
                    blocks = [blk.formats(['coo','csr','csc']) for blk in blocks]

                    for blk in blocks:
                        blk.create_formats_()
                e = []
                n = [blocks[0].num_src_nodes()]
                for blk in blocks:
                    edges_computed += blk.edges()[0].shape[0]
                    e.append(blk.edges()[0].shape[0])
                    n.append(blk.num_dst_nodes())
                print(device, e, n)    
                #print(edges_computed)
                t3 = time.time()
                batch_time[DATALOAD_START_TIME] = t3
                e0.record()
                #start = offsets[device][0]
                #end = offsets[device][1]
                #hit = torch.where((input_nodes > start) & (input_nodes < end))[0].shape[0]
                if True:
                    if (3 in offsets):
                        hit = torch.where(input_nodes < offsets[3])[0].shape[0]
                    else: 
                        hit = 0
                    missed = input_nodes.shape[0] - hit
                else:
                    hit = 0
                    missed = 0
                # Load the input features as well as output labels
                #torch.cuda.nvtx.range_push("fetch")
                for _ in range(1):
                    e0.record()
                    batch_inputs, batch_labels = load_subtensor(
                        nfeat, labels, seeds, input_nodes, device)
                    #torch.cuda.nvtx.range_pop()
                    #nfeat[input_nodes].to(device)
                    e1.record()
                    e1.synchronize()
                    #gpu_order = nfeat.feature_order[input_nodes]

                    #for i in range(4):
                    #    first,last = offsets[i]
                    #    print("cache hit", device, "from",  i ,  torch.sum((gpu_order >=first ) & (gpu_order < last))/gpu_order.shape[0])
                    #print("Cache miss", torch.sum(gpu_order > offsets[3][1]), input_nodes.shape)
                print("Data", (batch_inputs.shape[0] * batch_inputs.shape[1]  * 4)/(1024 ** 2),  "Bandwidth", (batch_inputs.shape[0] * batch_inputs.shape[1] * 4 )/ (1024 **3 * (e0.elapsed_time(e1)/1000)))
                batch_time[DATALOAD_END_TIME] = time.time()
                # Compute loss and prediction
                torch.cuda.nvtx.range_push("training {} {}".format(e,n))
                
                with torch.autograd.profiler.profile(enabled=(False), use_cuda=True, profile_memory = True) as prof:
                    e3.record()
                    batch_pred = model(blocks, batch_inputs)
                    loss = loss_fcn(batch_pred, batch_labels.flatten())
                    
                    e4.record()
                    loss.backward()
                    e5.record()
                
                e5.synchronize()
                batch_time[END_BACKWARD] = time.time()
                torch.cuda.nvtx.range_pop()
                optimizer.step()
                r = torch.cuda.memory.max_memory_allocated()
                t = torch.cuda.get_device_properties(0).total_memory
                print("Total used  memory", (r)/(1024 ** 3), "GB")
 
                batch_time[FORWARD_ELAPSED_EVENT_TIME] = e3.elapsed_time(e4)/1000
                batch_time[DATALOAD_ELAPSED_EVENT_TIME] = e0.elapsed_time(e1)/1000
                #print("sample time", t2-t1, t3-t2)
                #print("Time feature", device, e1.elapsed_time(e2)/1000)
                #print("Expected bandwidth", missed * nfeat.shape[1] * 4/ ((e1.elapsed_time(e2)/1000) * 1024 * 1024 * 1024), "GB device", rank, "cache rate", hit/(hit + missed))
                data_movement_cpu += ((missed * nfeat.shape[1] * 4)/(1024 * 1024))
                data_movement_gpu += ((hit * nfeat.shape[1] * 4)/ (1024 ** 2))
                minibatch_metrics.append(batch_time)
                if args.early_stopping and step ==5:
                    break
                step = step + 1
                torch.cuda.nvtx.range_pop()
                total_loss += loss.item()
                total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
        #        pbar.update(args.batch_size)
        except StopIteration:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
            pass
        print("EPOCH TIME",time.time() - epoch_start)
        epoch_available_memory.append(torch.cuda.max_memory_allocated())
        epoch_metrics.append(minibatch_metrics)
        epoch_time.append(time.time()-epoch_start)
        edges_per_epoch.append(edges_computed)
        data_movement_cpu_epoch.append(data_movement_cpu)
        data_movement_gpu_epoch.append(data_movement_gpu)
        sample_time_epoch.append(sample_time)
        #pbar.close()

        loss = total_loss / len(dataloader)
        approx_acc = total_correct / (len(dataloader) * args.batch_size)
        accuracy.append(approx_acc)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}, Epoch Time: {time.time() - tic:.4f}')
    if rank == 0 :
        #print(prof.key_averages().table(sort_by='cuda_time_total'))

        if epoch >= 10:
            val_acc, test_acc = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
            print(f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    with open('metrics{}.pkl'.format(rank), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(epoch_metrics, outp, pickle.HIGHEST_PROTOCOL)
    if rank == 0:
        print("accuracy:{}".format(accuracy[-1]))
        print("epoch_time:{}".format(average(epoch_time)))
    print("edges_per_epoch:{}".format(average(edges_per_epoch)))
    print("data moved :{}MB".format(average(data_movement_cpu_epoch)))
    print("Alt Sample time:{}s".format(average(sample_time_epoch)))
    # print("data moved GPU:{}MB".format(average(data_movement_gpu_epoch)))
    print("Memory used :{}GB".format(max(epoch_available_memory)/(1024 ** 3)))
    dist.destroy_process_group()

    return final_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--cache-size', type = str, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    #argparser.add_argument('--num-workers', type=int, default=4,
    #                       help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--sample-gpu', action='store_true')
    argparser.add_argument('--data', type=str, choices=('cpu', 'gpu', 'quiver', 'unified'), default = 'quiver')
    argparser.add_argument('--early-stopping', action = 'store_true')

    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load ogbn-products data
    #root = "/mnt/bigdata/sandeep/"
    #from utils import  get_process_graph
    dg_graph, partition_map, num_classes = get_dgl_graph(args.graph)
    data = dg_graph
    train_idx = torch.where(data.ndata.pop('train'))[0]
    val_idx = torch.where(data.ndata.pop('valid'))[0]
    test_idx = torch.where(data.ndata.pop('test'))[0]
    # if args.graph != "mag240M":
    #     val_idx = torch.where(data.ndata.pop('val_mask'))[0]
    #     test_idx = torch.where(data.ndata.pop('test_mask'))[0]
    # else: 
    #     val_idx = None
    #     test_idx = None    
    print("Train IDx mask", train_idx.dtype)
    graph = dg_graph
    labels = dg_graph.ndata.pop('labels')
    assert(torch.max(labels[train_idx]) + 1 <= num_classes)
    
    n_classes = num_classes
    feat = dg_graph.ndata.pop('features')
    feat.share_memory_()
    print(graph.edges()[1][:30])
    graph = graph.add_self_loop()
    ###################################
    #data = DglNodePropPredDataset(name=args.graph, root=root)
    #splitted_idx = data.get_idx_split()
    #train_idx, val_idx, test_idx = data.ndata['train_idx'], splitted_idx['valid'], splitted_idx['test']
    #graph, labels = data[0]
    #graph = graph.add_self_loop()
    #labels = labels[:, 0].to(device)

    # feat = graph.ndata.pop('feat')
    print("Attempting to pin memory")
    # if args.graph != "mag240M":
    #     # Pinning memory doubles up for mag causing an out of memory 
    #     feat = feat.pin_memory()
    print("memory pinn successful  ")
    #year = graph.ndata.pop('year')
    if args.data == 'cpu':
        nfeat = feat
        offsets = {}
        offsets[3] = 0
    elif args.data == 'gpu':
        nfeat = feat.to(device)
    elif args.data == 'quiver':
        quiver.init_p2p(device_list = [0,1,2,3])
        csr_topo = quiver.CSRTopo(th.stack(graph.edges('uv')))
        #csr_topo = None
        device_cache_size = args.cache_size
        print("cache size", args.cache_size)
        #if float(args.cache_per) > .25:
        if False:
            cache_policy = "device_replicate"
            last_node_stored = nfeat.shape[0]
        else:
            cache_policy = "p2p_clique_replicate"
            #for device in range(4):
            #    start = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].start)
            #    end = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].end)
            #    offsets[device] = (start,end)
        nfeat = quiver.Feature(rank=0, device_list=[0,1,2,3],
                               #device_cache_size="200M",
                               device_cache_size = device_cache_size,
                               cache_policy = cache_policy,
                               #cache_policy="device_replicate",
                               csr_topo=csr_topo)
        #feat = dg_graph.in_degrees().unflatten(0, (dg_graph.num_nodes(), 1)) * torch.ones(dg_graph.num_nodes(), 10, dtype = torch.float32)
        #print(feat.shape)
        print("Get quiver feature")
        nfeat.from_cpu_tensor(feat)
        dtype = feat.dtype
        del feat
        gc.collect()
        print(dtype)
        #if float(args.cache_per) == .25 :
        if False:
            if len(nfeat.clique_tensor_list) != 0:
                last_node_stored = nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[3].end
            else:
                last_node_stored = 0
        print("Using quiver feature")
        #print(nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device)
        offsets = {}
        #offsets[3] = last_node_stored
        # Temporary disable
        #for device in range(4):
        #    start = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].start)
        #    end = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].end)
        #    offsets[device] = (start,end)
        #    print(device, nfeat[start:end])
        #print(offsets)
    elif args.data == 'unified':
        from distutils.version import LooseVersion
        assert LooseVersion(dgl.__version__) >= LooseVersion('0.8.0'), \
            f'Current DGL version ({dgl.__version__}) does not support UnifiedTensor.'
        nfeat = dgl.contrib.UnifiedTensor(feat, device=device)
        offsets = {}
        offsets[3] = 0
    else:
        raise ValueError(f'Unsupported feature storage location {args.data}.')

    in_feats = nfeat.shape[1]
    #n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    #graph.create_formats_()
    # Pack data
    metrics_queue = mp.Queue(4)
    train_idx = train_idx[torch.randperm(train_idx.shape[0])]

    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, offsets, metrics_queue, dtype


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
    collected_metrics = []
    for i in range(4):
        with open("metrics{}.pkl".format(i), "rb") as input_file:
            cm = pickle.load(input_file)

        collected_metrics.append(cm)
    epoch_batch_sample, epoch_batch_graph, epoch_batch_feat_time, \
            epoch_batch_forward, epoch_batch_backward, \
            epoch_batch_loadtime, epoch_batch_totaltime = \
                compute_metrics(collected_metrics)
    print("sample_time:{}".format(epoch_batch_sample))
    print("movement data time:{}".format(epoch_batch_loadtime))
    print("movement graph:{}".format(epoch_batch_graph))
    print("movement feature:{}".format(epoch_batch_loadtime))
    print("forward time:{}".format(epoch_batch_forward))
    print("backward time:{}".format(epoch_batch_backward))
