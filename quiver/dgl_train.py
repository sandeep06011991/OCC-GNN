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
import multiprocessing
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from utils.utils import *
import pickle
import torch.autograd.profiler as profiler
# from torch.profiler import record_function, ProfilerActivity
from training.setup import * 
from training.measure import * 
from models import *
from training.loop import * 

def average(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    #Note:: Cross check for variance, could be a wrong way to go about.
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



# Entry point

def run(rank, world_size, args, features, shared_graph, sampler, labels, idx_split, \
            in_feat_dim, n_classes):
    torch.cuda.set_device(rank)
    # Unpack data
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size=4)

    device = rank
    torch.cuda.set_device(device)
    labels = labels.to(device)
    train_nid = idx_split["train"]
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
    
    setup_logger(rank, "dgl", args, train_nid,0)
    
    train_nid = train_nid.to(device)
    
    train_dataloader, valid_dataloader = \
        get_dataloader(rank, args, shared_graph, train_nid, sampler, idx_split)
    
    number_of_minibatches = train_nid.shape[0]/args.batch_size
    
    # Define model and optimizer
    if args.model == "GCN":
        model = SAGE(in_feat_dim, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    else:
        assert(args.model == "GAT")
        heads = 4
        model = GAT(in_feat_dim, args.num_hidden, \
                n_classes , heads, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.feat_gpu:
        features = features.to(device)
    train(rank, args, model, train_dataloader,\
             optimizer, features, labels, in_feat_dim, valid_dataloader)
    # Training loop
    best_val_acc = final_test_acc = 0

    dist.destroy_process_group()

    return final_test_acc

if __name__ == '__main__':
    mp.set_start_method('spawn')
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default = 0)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--early-stopping', action = 'store_true')
    argparser.add_argument('--test-graph',type = str)
    argparser.add_argument('--no-uva', action = 'store_true')
    argparser.add_argument('--feat-gpu', action = 'store_true')
    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    graph, partition_map, n_classes, idx_split = get_dgl_graph(args.graph)

    labels = graph.ndata.pop('labels')
    labels = labels.flatten().clone()
    labels = labels.type(torch.int64)
    features = graph.ndata.pop('features')
    in_feat_dim = features.shape[1]
    fanouts = [int(f) for f in args.fan_out.split(",")]
    
    graph = graph.add_self_loop()
    graph.create_formats_()
    shared_graph = graph.shared_memory("dglgraph")
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    del graph 
    import gc
    gc.collect()

    in_feats = features.shape[1]
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    world_size = 4
    mp.spawn(run, args=(world_size, args, features,\
        shared_graph, sampler, labels, idx_split,  in_feat_dim, n_classes), nprocs=world_size, daemon=True)
   