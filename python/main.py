import argparse
import argparse
import torch
import dgl
# Ensures all enviroments are running on the correct version
assert(dgl.__version__ == '0.8.2')
import time
import nvtx
from dgl.sampling import sample_neighbors
from models.dist_gcn import get_sage_distributed
from models.dist_gat import get_gat_distributed
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager
import torch.optim as optim
from cslicer import cslicer
from data import Bipartite, Sample, Gpu_Local_Sample
import numpy as np
import threading
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import Queue
import torch.distributed as dist
import os
from slicing import slice_producer
#os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/cslicer/"
import time
import inspect
from utils.shared_mem_manager import *
from slicing import *
from train import *

def work_producer(work_queue,training_nodes, batch_size,
    no_epochs, num_workers, deterministic):
    # todo:
    training_nodes = training_nodes.tolist()
    num_nodes = len(training_nodes)
    for epoch in range(no_epochs):
        i = 0
        if not deterministic:
            random.shuffle(training_nodes)
        while(i < num_nodes):
            work_queue.put(training_nodes[i:i+batch_size])
            i = i + batch_size
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
    fanout = [10,10,10]
    sm_filename_queue = mp.Queue(NUM_BUCKETS)
    sm_manager = SharedMemManager(sm_filename_queue)

    # Create main objects
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage, \
                    fanout, batch_size,  partition_map, deterministic = args.deterministic)
    storage_vector = []
    for i in range(4):
        storage_vector.append(mm.local_to_global_id[i].tolist())

    work_queue = mp.Queue(8)
    train_mask = dg_graph.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()

    print("Training nodes", train_nid.shape[0])
    if args.deterministic:
        train_nid = torch.arange(dg_graph.num_nodes())
        # When debugging at node level
        # train_nid = torch.arange(0,32)
    minibatches_per_epoch = int(len(train_nid)/minibatch_size)

    print("Training on num nodes = ",train_nid.shape)

    # global train_nid_list
    # train_nid_list= train_nid.tolist()
    queue_size =1
    no_worker_process = 1

    work_producer_process = mp.Process(target=(work_producer), \
                  args=(work_queue, train_nid, minibatch_size, no_epochs,\
                    no_worker_process, args.deterministic))
    work_producer_process.start()

    sample_queues = [mp.Queue(queue_size) for i in range(4)]

    # sample_queues = [mp.SimpleQueue() for i in range(4)]
    lock = torch.multiprocessing.Lock()

    slice_producer_processes = []

    for proc in range(no_worker_process):
        slice_producer_process = mp.Process(target=(slice_producer), \
                      args=(graph_name, work_queue, sample_queues[0], lock,\
                                storage_vector, args.deterministic,
                                proc, sm_filename_queue))
        slice_producer_process.start()
        slice_producer_processes.append(slice_producer_process)


    procs = []
    n_gpus = 4
    labels = dg_graph.ndata["labels"]
    labels.share_memory_()


    for proc_id in range(n_gpus):
        p = mp.Process(target=(run_trainer_process), \
                      args=(proc_id, n_gpus, sample_queues, minibatches_per_epoch \
                       , features, args, \
                       num_classes, mm.batch_in[proc_id], labels,no_worker_process, args.deterministic,\
                        dg_graph.in_degrees(), sm_filename_queue, mm.local_sizes[proc_id],cache_percentage))
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
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv")
    # training details
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num-workers', type=int, default=4,
       help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--fsize', type = int, default = -1, help = "use only for synthetic")
    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-per', type =float, default = .25)
    argparser.add_argument('--model',help="gcn|gat")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=3,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=(4096))
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--deterministic', default = False, action="store_true")
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    mp.set_start_method('spawn')
    assert(args.model in ["gcn","gat"])
    main(args)
