import argparse
import argparse
import torch
import dgl
# Ensures all enviroments are running on the correct version
assert(dgl.__version__ == '0.9.1post1' or dgl.__version__ == '0.8.2' or dgl.__version__=='0.9.1')
import time
import nvtx
from dgl.sampling import sample_neighbors
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
from cslicer import cslicer
from slicing import slice_producer

#os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/cslicer/"
import time
import inspect
from utils.shared_mem_manager import *
from slicing import *
from train import *

def work_producer(work_queue,training_nodes, batch_size,
    no_epochs, num_workers, deterministic, early_stopping):
    # todo:
    training_nodes = training_nodes.tolist()
    num_nodes = len(training_nodes)
    for epoch in range(no_epochs):
        i = 0
        step = 0
        if not deterministic:
            random.shuffle(training_nodes)
        while(i < num_nodes):
            work_queue.put(training_nodes[i:i+batch_size])
            i = i + batch_size
            if early_stopping and step == 5:
                break
            #print(step, epoch)
            step  = step  + 1
        work_queue.put("EPOCH")
    for n in range(num_workers):
        work_queue.put("END")
    while(True):
        if(work_queue.qsize()==0):
            break
        time.sleep(1)
    time.sleep(30)
    print("WORK PRODUCER TRIGGERING END")


def main(args):
    print("Graph read pk")
    graph_name = args.graph
    dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize)
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
    minibatch_size = batch_size
    fanout = args.fan_out.split(',')
    fanout = [(int(f)) for f in fanout]
    for i in fanout:
        assert(i == fanout[0])
        # Onlysupport same fanout currently.

    fanout_val = fanout[0]
    import random
    file_id = random.randint(0,10000)

    sm_filename_queue = mp.Queue(get_number_buckets(args.num_workers))
    sm_manager = SharedMemManager(sm_filename_queue, args.num_workers, file_id)

    #Not applicable as some nodes have zero features in ogbn-products
    #assert(not torch.any(torch.sum(features,1)==0))

    # Create main objects
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage, \
                    fanout, batch_size,  partition_map, deterministic = args.deterministic)
    storage_vector = []
    for i in range(4):
        storage_vector.append(mm.local_to_global_id[i].tolist())
    print("memory manaager done")
    work_queue = mp.Queue(8)
    train_mask = dg_graph.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()

    print("Training nodes", train_nid.shape[0])
    if args.deterministic:
        train_nid = torch.arange(dg_graph.num_nodes())
        # When debugging at node level
        # train_nid = torch.arange(0,32)
    minibatches_per_epoch = int(len(train_nid)/minibatch_size)
    if args.model == "gcn":
        self_edge = False
    if args.model == "gat":
        self_edge = True
    print("Training on num nodes = ",train_nid.shape)
    # global train_nid_list
    # train_nid_list= train_nid.tolist()
    queue_size = args.num_workers
    no_worker_process = args.num_workers

    work_producer_process = mp.Process(target=(work_producer), \
                  args=(work_queue, train_nid, minibatch_size, no_epochs,\
                    no_worker_process, args.deterministic, args.early_stopping))
    work_producer_process.start()

    sample_queues = [mp.Queue(queue_size) for i in range(4)]

    # sample_queues = [mp.SimpleQueue() for i in range(4)]
    
    lock = torch.multiprocessing.Lock()

    slice_producer_processes = []
    print("Start slicer weorkers")
    for proc in range(no_worker_process):
        slice_producer_process = mp.Process(target=(slice_producer), \
                      args=(graph_name, work_queue, sample_queues[0], lock,\
                                storage_vector, args.deterministic,
                                proc, sm_filename_queue, no_worker_process, fanout_val, file_id, self_edge))
        slice_producer_process.start()
        slice_producer_processes.append(slice_producer_process)


    procs = []
    n_gpus = 4
    labels = dg_graph.ndata["labels"]
    labels.share_memory_()

    print("Trainer processes")
    for proc_id in range(n_gpus):
        p = mp.Process(target=(run_trainer_process), \
                      args=(proc_id, n_gpus, sample_queues, minibatches_per_epoch \
                       , features, args, \
                       num_classes, mm.batch_in[proc_id], labels,no_worker_process, args.deterministic,\
                        sm_filename_queue, mm.local_sizes[proc_id],cache_percentage, file_id))
        p.start()
        procs.append(p)
    for sp in slice_producer_processes:
        sp.join()
    print("Sampler returned")
    for p in procs:
        p.join()
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
    argparser.add_argument('--num-workers', type=int, default=16,
       help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--fsize', type = int, default = -1, help = "use only for synthetic")
    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-per', type =float, default = .25, required = True)
    argparser.add_argument('--model',help="gcn|gat", required = True)
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=3,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=(4096))
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--deterministic', default = False, action="store_true")
    argparser.add_argument('--early-stopping', action = "store_true")
    argparser.add_argument('--test-graph-dir', type = str)
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    mp.set_start_method('spawn')
    assert(args.model in ["gcn","gat"])
    main(args)
