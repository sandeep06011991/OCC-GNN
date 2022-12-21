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

from torch.profiler import profile, record_function, ProfilerActivity
from dgl_sage import SAGE
from dgl_gat import GAT
from utils.async_timing_analysis import *
from utils.env import *
from utils.utils import *
import pickle

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
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# Entry point
import logging,os

def run(rank, args,  data):
    # Unpack data
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size=4)
    
    if rank == 0:
      os.makedirs('{}/quiver/logs_quiver'.format(ROOT_DIR),exist_ok = True)
      FILENAME= ('{}/quiver/logs_quiver/{}_{}_{}.txt'.format(ROOT_DIR, \
               args.graph, args.batch_size, args.model))

      fileh = logging.FileHandler(FILENAME, 'w')
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      fileh.setFormatter(formatter)
      log = logging.getLogger()  # root logger
      log.addHandler(fileh)      # set the new handler
      log.setLevel(logging.INFO)

    
    device = rank
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, offsets, metrics_queue, feat_cpu, offsets, local_order = data
    assert(feat_cpu.device == torch.device('cpu'))
    if args.data == 'gpu':
        nfeat = nfeat.to(rank)
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        g = g.formats(['csc'])
        g = g.to(device)
    print("Cross check edges moved !")
    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')], replace = True)
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    world_size = 4
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
    
    number_of_minibatches = train_nid.shape[0]/args.batch_size

    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor = 2,
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
    data_movement_cpu_epoch = []
    data_movement_gpu_epoch = []
    epoch_metrics = []
    accuracy = []
    edges_per_epoch = []
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
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
        dataloader_i = iter(dataloader)
        edges_computed = 0
        data_movement_cpu = 0
        data_movement_gpu = 0
        global_cache_misses = 0
        step = 0
        minibatch_metrics = []
        try:
            while True:
                batch_time = {}
                optimizer.zero_grad()
                t1 = time.time()
                input_nodes, seeds, blocks = next(dataloader_i)
                t2 = time.time()
                batch_time[SAMPLE_START_TIME] = t1
                batch_time[GRAPH_LOAD_START_TIME] = t2
                # copy block to gpu
                blocks = [blk.to(device) for blk in blocks]
                #t2 = time.time()
                blocks = [blk.formats(['coo','csr','csc']) for blk in blocks]

                for blk in blocks:
                    blk.create_formats_()
                #print(edges_computed)
                t3 = time.time()
                batch_time[DATALOAD_START_TIME] = t3
                e0.record()
                #start = offsets[device][0]
                #end = offsets[device][1]
                #hit = torch.where((input_nodes > start) & (input_nodes < end))[0].shape[0]
                ########################################################
                #cpu_pull = torch.where(local_order[input_nodes]>offsets[3]) 
                cpu_hit = torch.where(local_order[input_nodes] > offsets[4])[0].shape[0]
                gpu_hit = torch.where(torch.logical_and((local_order[input_nodes] >= offsets[rank]),(local_order[input_nodes] < offsets[rank + 1])))[0].shape[0]
                gpu_hit = input_nodes.shape[0] - cpu_hit - gpu_hit
                hits = {}
                s = 0
                for i in range(4):
                    hits[i] = torch.where(torch.logical_and((local_order[input_nodes] >= offsets[i]),(local_order[input_nodes] < offsets[i + 1])))[0].shape[0]
                    s += hits[i]
                
                print("Local hit rate", rank, hits, s, input_nodes.shape[0])
                # Load the input features as well as output labels
                torch.distributed.barrier() 
                torch.cuda.synchronize()
                batch_inputs, batch_labels = load_subtensor(
                        nfeat, labels, seeds, input_nodes, device)
                torch.cuda.synchronize()
                torch.distributed.barrier() 
                #t2 = time.time()
                #torch.cuda.synchronize()
                #t3 = time.time()   
                #print("Execpt a >b", t2-t1, t3-t2)
                e1.record()
                batch_time[DATALOAD_END_TIME] = time.time()
                # Compute loss and prediction
                #print("Torch sum", torch.sum(batch_inputs))
                #torch.distributed.barrier()
                for i in range(4):
                    torch.cuda.set_device(i)
                    torch.cuda.synchronize()
                torch.cuda.set_device(rank)    
                #batch_inputs = feat_cpu[input_nodes].to(device)
                #prevent leakage
                #with profile( activities=[
        #ProfilerActivity.CPU, ProfilerActivity.CUDA],  use_cuda=True, profile_memory = True) as prof:
                if True:
                #print("quiver",batch_inputs.is_contiguous())
                    
                    e3.record()
                    batch_pred = model(blocks, batch_inputs)
                    e4.record()
                    #print("quiver time", e3.elapsed_time(e4)/1000, device)
                    #batch_inputs = feat_cpu[input_nodes].to(device)
                    #e3.record()
                    
                    #batch_pred = model(blocks , batch_inputs)
                    #e4.record()
                    #e4.synchronize()
                    #print("Naive", batch_inputs.is_contiguous())
                   #print("NAvie time", e3.elapsed_time(e4)/1000, device)
                #print("forward time", e3.elapsed_time(e4)/1000)

                loss = loss_fcn(batch_pred, batch_labels)
                loss.backward()
                batch_time[END_BACKWARD] = time.time()
                #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                e4.synchronize()
                optimizer.step()
                batch_time[FORWARD_ELAPSED_EVENT_TIME] = e3.elapsed_time(e4)/1000
                batch_time[DATALOAD_ELAPSED_EVENT_TIME] = e0.elapsed_time(e1)/1000
                #torch.cuda.synchronize()
                #print("sample time", t2-t1, t3-t2)
                #print("Time feature", device, e1.elapsed_time(e2)/1000)
                #print("Expected bandwidth", missed * nfeat.shape[1] * 4/ ((e1.elapsed_time(e2)/1000) * 1024 * 1024 * 1024), "GB device", rank, "cache rate", hit/(hit + missed))
                data_movement_cpu += (cpu_hit * nfeat.shape[1] * 4/(1024 * 1024))
                data_movement_gpu += (gpu_hit * nfeat.shape[1] * 4/(1024 * 1024))
                minibatch_metrics.append(batch_time)
                if args.early_stopping and step ==20:
                    break
                step = step + 1
                print("forward time", e3.elapsed_time(e4)/1000)
                #print("backward time", e3.elapsed_time(e4)/1000)
                total_loss += loss.item()
                total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
        #        pbar.update(args.batch_size)
                if rank == 0:
                    log.info("step {}, epoch {}, number_of_minibatches {}".format(step, epoch, number_of_minibatches))

        except StopIteration:
            pass
        print("EPOCH TIME",time.time() - epoch_start)
        epoch_metrics.append(minibatch_metrics)
        epoch_time.append(time.time()-epoch_start)
        edges_per_epoch.append(edges_computed)
        data_movement_cpu_epoch.append(data_movement_cpu)
        data_movement_gpu_epoch.append(data_movement_gpu)
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
    print("data moved CPU:{}MB".format(average(data_movement_cpu_epoch)))
    print("data moved GPU:{}MB".format(average(data_movement_gpu_epoch)))
    dist.destroy_process_group()

    return final_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--cache-per', type =float, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
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
    dg_graph, partition_map, num_classes = get_process_graph(args.graph, -1)
    data = dg_graph
    train_idx = torch.where(data.ndata.pop('train_mask'))[0]
    val_idx = torch.where(data.ndata.pop('val_mask'))[0]
    test_idx = torch.where(data.ndata.pop('test_mask'))[0]
    graph = dg_graph
    labels = dg_graph.ndata.pop('labels')
    feat = dg_graph.ndata.pop('features')
    
    graph = graph.add_self_loop()
    ###################################
    #data = DglNodePropPredDataset(name=args.graph, root=root)
    #splitted_idx = data.get_idx_split()
    #train_idx, val_idx, test_idx = data.ndata['train_idx'], splitted_idx['valid'], splitted_idx['test']
    #graph, labels = data[0]
    #graph = graph.add_self_loop()
    #labels = labels[:, 0].to(device)

    # feat = graph.ndata.pop('feat')
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
        cache_size = int(float(args.cache_per) * feat.shape[0] * feat.shape[1] * 4/(1024 * 1024))
        device_cache_size = "{}M".format(cache_size)
        print("calculated cache_size ", cache_size)
        if float(args.cache_per) > .25:
            cache_policy = "device_replicate"
            last_node_stored = nfeat.shape[0]
        else:
            cache_policy = "p2p_clique_replicate"
            #for device in range(4):
            #    start = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].start)
            #    end = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].end)
            #    offsets[device] = (start,end)
        #
        #print("CSR topo is none")
        #csr_topo = None
        nfeat = quiver.Feature(rank=0, device_list=[0,1,2,3],
                               #device_cache_size="200M",
                               device_cache_size = device_cache_size,
                               cache_policy = cache_policy,
                               #cache_policy="device_replicate",
                               csr_topo=csr_topo)
        nfeat.from_cpu_tensor(feat)
        if float(args.cache_per) <= .25:
            print(nfeat.clique_tensor_list, "Check")
            print(nfeat.clique_tensor_list[0].shard_tensor_config,"jksdad")
            if len(nfeat.clique_tensor_list) != 0:
                try:
                    last_node_stored = nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[3].end
                except KeyError:
                    last_node_stored = 0
            else:
                last_node_stored = 0
        
        local_orders = nfeat.feature_order.clone()
        offsets = [0] 
        shard_tensor = nfeat.clique_tensor_list[0]
        offset_config = shard_tensor.shard_tensor_config.tensor_offset_device
        for d in range(4):
            if d in offset_config.keys():
                offsets.append(offset_config[d].end_)
            else:
                offsets.append(0)
                #shard_tensor.shard_tensor_config.tensor_offset_device[d]
            #print("offset", sv.start_, sv.end_)

        #offsets = {}
        #offsets[3] = last_node_stored
        # Temporary disable
        #for device in range(4):
        #    start = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].start)
        #    end = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].end)
        #    offsets[device] = (start,end)
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
    metrics_queue = mp.Queue(4)
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, offsets, metrics_queue, feat, offsets, local_orders


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
    print("forward time:{}".format(abs(epoch_batch_forward)))
    print("backward time:{}".format(abs(epoch_batch_backward)))
