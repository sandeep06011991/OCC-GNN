import argparse
import argparse
import torch
import dgl
# Ensures all enviroments are running on the correct version
assert(dgl.__version__ == '0.9.1post1' or\
        dgl.__version__ == '0.8.2' or dgl.__version__ == '0.8.2post1' or dgl.__version__=='0.9.1')

import time
import nvtx
from models.dist_gcn import get_sage_distributed
from models.dist_gat import get_gat_distributed
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
import inspect
from cu_shared import *

# Trainer Queues, assign work to each of the trainer which are later shuffled
# deterministic flag disables shuffle for e2e testing to with naive dgl version
def work_producer(trainer_queues,training_nodes, batch_size,
    no_epochs, num_workers, deterministic = False):
    # TODO: num_workers is redundant
    assert(len(trainer_queues) == num_workers)
    # Problem one worker could probably not see it.
    training_nodes = training_nodes.tolist()
    num_nodes = len(training_nodes)
    deterministic = False
    for epoch in range(no_epochs):
        current_offset = 0
        step = 0
        if not deterministic:
            random.shuffle(training_nodes)
        # batches across epochs are not distinguised as multiple queues are not synchronized and do not gurantee FIFO
        while(current_offset +  batch_size < num_nodes):
            for j in range(num_workers):
                if current_offset + batch_size < num_nodes:
                    trainer_queues[j].put(training_nodes[current_offset:current_offset + batch_size])
                   
                else:
                    trainer_queues[j].put("DUMMY")
                current_offset = current_offset + batch_size
    print("All samples generated")
    # for j in range(num_workers):
    #     trainer_queues[j].put("END")
    print("Waiting for clean up")
    while(True):
        all_empty = True
        for j in range(num_workers):
            if(trainer_queues[j].qsize()!=0):
                all_empty = False
            time.sleep(1)
        if all_empty:
            break
    time.sleep(1)

    print("WORK PRODUCER TRIGGERING END")


def main(args):
    graph_name = args.graph
    if args.random_partition:
        assert(args.num_gpus == 4)
        dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize, -1)
    else:
        dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize, args.num_gpus)
    partition_map = partition_map.type(torch.LongTensor)
    
    if not args.deterministic:
        features = dg_graph.ndata["features"]
        features = features.pin_memory()
    else:
        features = dg_graph.ndata["features"]
        num_nodes = features.shape[0]
        features = torch.arange(0,num_nodes).reshape(num_nodes,1)
    features.share_memory_()
    cache_percentage = args.cache_per
    batch_size = args.batch_size
    no_epochs = args.num_epochs
    num_gpus = args.num_gpus
    minibatch_size = batch_size
    fanout = args.fan_out.split(',')
    fanout = [(int(f)) for f in fanout]

    total_minibatches = int(features.shape[0]/batch_size) * no_epochs

    import random
    file_id = random.randint(0,10000)
    sm_manager = SharedMemManager(num_gpus, file_id)

    #Not applicable as some nodes have zero features in ogbn-products
    #assert(not torch.any(torch.sum(features,1)==0))
    # Create main objects
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage, \
                    fanout, batch_size,  partition_map, num_gpus, deterministic = args.deterministic)
    storage_vector = []
    num_workers = num_gpus
    for i in range(num_gpus):
        storage_vector.append(mm.local_to_global_id[i].tolist())
        print("STorage check",len(storage_vector[-1]))
    # Each gpu gets vertices to sample from this queue from work producer
    work_queues = [mp.Queue(3) for _ in range(num_workers)]
    # Exchange meta data required to read from shared memory

    exchange_queue = [mp.Queue(num_workers) for _ in range(num_workers)]
    train_mask = dg_graph.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()
    for i in range(args.num_gpus):
        print("Nodes per partition", torch.sum(partition_map[train_nid]  == i))
        print("Total degree", torch.sum(dg_graph.in_degrees(torch.where(partition_map[train_nid] == i)[0])))
    print("Training nodes", train_nid.shape[0])
    if args.deterministic:
        train_nid = torch.arange(dg_graph.num_nodes())

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

    shm = None
    if args.reusable_buffers:
        shm = []
        for i in range(num_gpus):
            i_shm = []
            for j in range(num_gpus):
                t = torch.ones((10000*1000), device = i)
                t.share_memory_()
                i_shm.append(t)
            shm.append(i_shm)



    work_producer_process = mp.Process(target=(work_producer), \
                  args=(work_queues, train_nid, minibatch_size, no_epochs,\
                    num_workers))
    work_producer_process.start()
    #print("Change to make sampling not a bottleneck")
    #sample_queues = [mp.Queue(queue_size) for i in range(4)]
    procs = []
    labels = dg_graph.ndata["labels"]
    labels.share_memory_()
    barrier = mp.Barrier(num_gpus)
    if args.optimization1 :
        from cu_train_opt import run_trainer_process
    else:
        from cu_train import run_trainer_process
    for proc_id in range(num_gpus):
        p = mp.Process(target=(run_trainer_process), \
                      args=(proc_id, num_gpus, work_queues[proc_id], minibatches_per_epoch \
                       , features, args, \
                       num_classes, mm.batch_in[proc_id], labels, \
                         args.deterministic,\
                         mm.local_sizes[proc_id],cache_percentage, file_id, args.num_epochs,\
                            storage_vector, fanout, exchange_queue, args.graph, args.num_layers, num_gpus, shm, barrier))
        p.start()
        procs.append(p)
    for proc in procs:
        proc.join()
    print("All Sampler returned")
    work_producer_process.join()
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
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    # .007 is best
    argparser.add_argument('--fsize', type = int, default = -1, help = "use only for synthetic")
    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-per', type =float, default = .25, required = True)
    argparser.add_argument('--model',help="gcn|gat", required = True)
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=(4096), required = True)
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--deterministic', default = False, action="store_true")
    argparser.add_argument('--early-stopping', action = "store_true")
    argparser.add_argument('--test-graph-dir', type = str)
    argparser.add_argument('--num-gpus', type = int, required = True)
    argparser.add_argument('--random-partition', action = "store_true", default = False)
    argparser.add_argument('--optimization1', action = "store_true", default = False)
    argparser.add_argument('--skip-shuffle', action = "store_true", default = False)
    argparser.add_argument('--load-balance', action = "store_true", default = False)
    argparser.add_argument('--barrier', action = "store_true", default = False)
    argparser.add_argument('--reusable-buffers', action = "store_true", default = False)
    # We perform only transductive training

    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    mp.set_start_method('spawn')
    assert(args.model in ["gcn","gat", "gat-pull"])
    main(args)
