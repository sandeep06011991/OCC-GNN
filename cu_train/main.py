import argparse
import argparse
import torch
import dgl
assert(dgl.__version__ == '1.1.0+cu118')
import time
import nvtx
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager
import torch.optim as optim
from data import Bipartite, Sample, Gpu_Local_Sample
import numpy as np
import threading
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import Queue
import torch.distributed as dist
import os
import pwd
import sys
'''
uname = pwd.getpwuid(os.getuid())[0]
if uname == 'spolisetty':
    ROOT_DIR = "/home/spolisetty/OCC-GNN/cslicer/"
    SRC_DIR = "/home/spolisetty/OCC-GNN/python/main.py"
    SYSTEM = 'jupiter'
    OUT_DIR = '/home/spolisetty/OCC-GNN/experiments/exp6/'
if uname == 'q91':
    ROOT_DIR = "/home/q91/OCC-GNN/cslicer/"
    SRC_DIR = "/home/q91/OCC-GNN/python/main.py"
    SYSTEM = 'ornl'
    OUT_DIR = '/home/q91/OCC-GNN/experiments/exp6/'
if uname == 'spolisetty_umass_edu':
    ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/cslicer"
    SRC_DIR = "/home/spolisetty_umass_edu/OCC-GNN/python/main.py"
    SYSTEM = 'unity'
    OUT_DIR = '/home/spolisetty_umass_edu/OCC-GNN/experiments/exp6/'



path_set = False
for p in sys.path:
    print(p)
    if ROOT_DIR ==  p:
       path_set = True
if (not path_set):
    print("Setting Path")
    sys.path.append(ROOT_DIR)
'''
from utils.utils import *
import time
from cu_shared import *


def main(args):
    graph_name = args.graph
    if args.random_partition:
        assert(args.num_gpus == 4)
        dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize, -1)
    else:
        dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize, args.num_gpus)
    print("Read all data")
    features = dg_graph.ndata.pop('features')
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    valid_nid = dg_graph.ndata['val_mask'].nonzero().flatten()
    if True:
        print("Taking in features")
        if graph_name != "mag240M":
            print("Pin memory")
            assert(features.is_shared())
            features = features.share_memory_()
            
    else:
        features = dg_graph.ndata["features"]
        num_nodes = features.shape[0]
        features = torch.arange(0,num_nodes).reshape(num_nodes,1)
    cache_size = args.cache_size
    batch_size = args.batch_size
    no_epochs = args.num_epochs
    num_gpus = args.num_gpus
    minibatch_size = batch_size
    fanout = args.fan_out.split(',')
    fanout = [(int(f)) for f in fanout]

    total_minibatches = int(features.shape[0]/batch_size) * no_epochs
    assert(features.is_shared())
    import random

    num_workers = num_gpus
    # for i in range(num_gpus):
    #     storage_vector.append(mm.local_to_global_id[i].tolist())
    #     print("STorage check",len(storage_vector[-1]))
    # Each gpu gets vertices to sample from this queue from work producer
    # Exchange meta data required to read from shared memory

    exchange_queue = [mp.Queue(num_workers) for _ in range(num_workers)]
    train_mask = dg_graph.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()
    for i in range(args.num_gpus):
        print("Nodes per partition", partition_map[i + 1] - partition_map[i])
        print("Total degree", torch.sum(dg_graph.in_degrees()[partition_map[i] : partition_map[i + 1]].to(torch.int32)))
    print("Training nodes", train_nid.shape[0])

    minibatches_per_epoch = int(len(train_nid)/minibatch_size)
    pull_optimization = False
    rounds = 0
    if args.model == "gcn":
        self_edge = False
    if args.model == "gat":
        self_edge = True
    if args.model == "gat-pull":
        self_edge = True
        pull_optimization = True
        rounds = 4
    val_acc_queue = mp.Queue(4)
    procs = []
    labels = dg_graph.ndata["labels"]
    labels.share_memory_()
    from cu_train_opt import run_trainer_process
    for proc_id in range(num_gpus):
        print("Starting ", proc_id)
        assert(features.is_shared())
        p = mp.Process(target=(run_trainer_process), \
                      args=(proc_id, num_gpus,  features, args, \
                       num_classes, labels, \
                         cache_size, \
                          fanout, exchange_queue, \
                              args.graph, args.num_layers, train_nid, valid_nid,\
                                val_acc_queue))
        p.start()
        procs.append(p)

    for proc in procs:
        proc.join()
    print("All Sampler returned")
    print("Leader process returns")
    #for p in procs:
     #   print("Waiting for trainer process")
     #   p.join()
      #  print("Trainer processes returning")
    time.sleep(30)
    for p in procs:
        p.terminate()
    print("Setup Done")


if __name__=="__main__":
    # python3 main.py --model gcn --deterministic
    # Unit test complete flow.
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv", required = True)
    # training details
    argparser.add_argument('--lr', type=float, default=0.003)
    # .007 is best
    argparser.add_argument('--fsize', type = int, default = -1, help = "use only for synthetic")
    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-size', type =str,  required = True)
    argparser.add_argument('--model',help="gcn|gat", required = True)
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=(4096), required = True)
    argparser.add_argument('--test-graph-dir', type = str)
    argparser.add_argument('--num-gpus', type = int, required = True)
    argparser.add_argument('--random-partition', action = "store_true", default = False)
    argparser.add_argument('--skip-shuffle', action = "store_true", default = False)
    argparser.add_argument('--load-balance', action = "store_true", default = False)
    argparser.add_argument('--use-uva', action = "store_true", default = False)
    # We perform only transductive training
    
    args = argparser.parse_args()
    mp.set_start_method('spawn')
    assert(args.model in ["gcn","gat", "gat-pull"])
    main(args)
