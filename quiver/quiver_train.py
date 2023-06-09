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
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from models import *
from utils.data.env import *
from utils.utils import *
from training.measure import *

import models
import pickle
import nvtx
import gc
from training.setup import * 
from training.loop import * 

def average(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)

import quiver

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)



def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# Entry point
def run(rank, world_size, args, qfeat, shared_graph, sampler, labels, idx_split, \
          last_node_stored: int, in_feat_dim, n_classes):
    torch.cuda.set_device(rank)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size= world_size)
    device = rank

    labels = labels.to(device)
    train_nid = idx_split["train"]
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
    
    setup_logger(rank, "quiver", args, train_nid, args.cache_size)    
    train_nid = train_nid.to(device)
    train_dataloader, valid_dataloader = \
        get_dataloader(rank, args, shared_graph, train_nid, sampler, idx_split)
    
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
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_val_acc = final_test_acc = 0

    train(rank, args, model, train_dataloader,\
                optimizer, qfeat, labels, in_feat_dim, valid_dataloader, last_node_stored = last_node_stored)
    dist.destroy_process_group()

    return final_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--cache-size', type = str, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dropout', type = float, default = .1)
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--early-stopping', action = 'store_true')
    argparser.add_argument('--no-uva', action = 'store_true')
    argparser.add_argument('--wd', type=float, default=0)

    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]

    graph, partition_map, n_classes, idx_split = get_dgl_graph(args.graph)

    labels = graph.ndata.pop('labels')
    labels = labels.flatten().clone()
    labels = labels.type(torch.int64)
    features = graph.ndata.pop('features')
    in_feat_dim = features.shape[1]
    fanouts = [int(f) for f in args.fan_out.split(",")]
    
    graph = graph.add_self_loop()
    row, col = graph.adj_tensors(fmt = "coo")
    #graph.create_formats_()
    shared_graph = graph.shared_memory("dglgraph")
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    del graph 
    gc.collect()

    csr_topo = quiver.CSRTopo(edge_index=(row, col))
    world_size = 4
    quiver.init_p2p(device_list=list(range(world_size)))
    qfeat = quiver.Feature(0, device_list=list(range(world_size)), cache_policy="p2p_clique_replicate", \
            device_cache_size=  args.cache_size, csr_topo = csr_topo)
    qfeat.from_cpu_tensor(features)

    try:
        last_node_stored = qfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[3].end
    except:
        last_node_stored = 0
    del features
    gc.collect()
    
    mp.spawn(run, args=(world_size, args, qfeat,\
        shared_graph, sampler, labels, idx_split, last_node_stored, in_feat_dim, n_classes), nprocs=world_size, daemon=True)
   
