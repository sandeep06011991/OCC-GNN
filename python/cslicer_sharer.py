import argparse
import torch
import dgl
import time
import nvtx
from dgl.sampling import sample_neighbors
from models.factory import get_model_distributed
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager
from utils.sampler import Sampler
import torch.optim as optim
from cslicer import cslicer
from data.cpu_compr_bipartite import Bipartite, Sample, Gpu_Local_Sample
import numpy as np

import threading
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import Queue
from queue import Queue as Simple_Queue
import threading
import os
os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/cslicer/"
import time

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

seed = 0
# set_all_seeds(seed)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum(),len(pred )


def run_trainer_process(proc_id, gpus, sample_queue, minibatches_per_epoch, features, args\
                    ,num_classes,batch_in, labels, num_sampler_workers, deterministic):
    print("Num sampler workers ", num_sampler_workers)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = gpus
    torch.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    if deterministic:
        set_all_seeds(seed)
    model = get_model_distributed(args.num_hidden, features, num_classes, proc_id, args.deterministic)
    model = model.to(proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id],\
                output_device = proc_id, find_unused_parameters=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
    labels= labels.to(proc_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    i = 0
    sample_get_time_epoch = []
    forward_time_epoch = []
    backward_time_epoch = []
    sample_move_time_epoch = []
    epoch_time = []
    fp_start = torch.cuda.Event(enable_timing=True)
    fp_end = torch.cuda.Event(enable_timing=True)
    bp_end = torch.cuda.Event(enable_timing=True)

    sample_get_time = 0
    forward_time = 0
    backward_time = 0
    sample_move_time = 0
    torch.cuda.set_device(proc_id)
    nmb = 0
    ii = 0
    t1 = time.time()
    print("Features ", features.device)
    num_epochs = 0
    while(True):
        t11 = time.time()
        nmb += 1
        t = sample_queue.get()
        if(type(t) == type("")):
            gpu_local_sample = t
        else:
            gpu_local_sample = Gpu_Local_Sample.deserialize(t)
        t22 = time.time()
        sample_get_time += t22 - t11
        if(gpu_local_sample == "EPOCH"):
            t2 = time.time()
            optimizer.zero_grad()
            sample_get_time_epoch.append(sample_get_time)
            forward_time_epoch.append(forward_time)
            backward_time_epoch.append(backward_time)
            sample_move_time_epoch.append(sample_move_time)
            epoch_time.append(t2-t1)
            sample_get_time = 0
            forward_time = 0
            backward_time = 0
            t1 = time.time()
            num_epochs += 1
            continue
        if(gpu_local_sample == "END"):
            print("GOT END OF FLAG")
            num_sampler_workers -= 1
            print("got end of epoch flag")
            # Maintain num active sampler workers
            if num_sampler_workers == 0:
                break
            else:
                continue
        t33 = time.time()
        gpu_local_sample.to_gpu()
        # gpu_local_sample.debug()
        t44 = time.time()
        sample_move_time += t44 - t33

        classes = labels[gpu_local_sample.last_layer_nodes].to(torch.device(proc_id))
        # print("Last layer nodes",gpu_local_sample.last_layer_nodes)
        # with nvtx.annotate("forward",color="blue"):
        torch.cuda.set_device(proc_id)
        fp_start.record()

        optimizer.zero_grad()
        output = model.forward(gpu_local_sample,batch_in, features)

        torch.cuda.set_device(proc_id)
        fp_end.record()
        loss = loss_fn(output,classes)/args.batch_size
        # print("loss",loss)
        loss.backward()
        # Helpful trick to make manual calculation of gradients easy.
        # torch.sum(output).backward()
        #
        for p in model.parameters():
            p.grad *= 4
        if deterministic:
            model.module.print_grad()
        ii = ii + 1
        bp_end.record()

        if True  and output.shape[0] !=0:
            acc = compute_acc(output,classes)
            print("accuracy {}",acc)
        torch.cuda.synchronize(bp_end)
        forward_time += fp_start.elapsed_time(fp_end)/1000
        # with nvtx.annotate("backward", color="red"):
        backward_time += fp_end.elapsed_time(bp_end)/1000
        if deterministic:
            model.module.print_grad()
        torch.distributed.barrier()

        optimizer.step()


    dev_id = proc_id
    if proc_id == 0:
        print("avg forward time: {}sec, device {}".format(sum(forward_time_epoch[1:])/(num_epochs - 1), dev_id))
        print(forward_time_epoch)
        print("avg backward time: {}sec, device {}".format(sum(backward_time_epoch[1:])/(num_epochs - 1), dev_id))
        print("avg move time: {}sec, device {}".format(sum(sample_move_time_epoch[1:])/(num_epochs - 1), dev_id))
        print(sample_move_time)
        print('avg epoch time: {}sec, device {}'.format(sum(epoch_time[1:])/(num_epochs - 1), dev_id))
        print(epoch_time)
        print('avg sample get time:{}sec, device {}'.format(sum(sample_get_time_epoch[1:])/(num_epochs - 1),dev_id))
        print(sample_get_time_epoch)
    # print("Memory",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
    # print("Thread running")

def work_producer(work_queue,training_nodes, batch_size, no_epochs, num_workers, deterministic):
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

def slice_producer(graph_name, work_queue, sample_queues, \
    lock , minibatches_per_epoch, no_epochs, storage_vector, deterministic):
    queue_size = 1
    no_worker_threads = 1
    sampler = cslicer(graph_name, queue_size, no_worker_threads,
            no_epochs, minibatches_per_epoch, storage_vector,deterministic)

    # Todo clean up unnecessary iterations
    while(True):
        sample_nodes = work_queue.get()
        if((sample_nodes) == "END"):
            lock.acquire()
            for qid,q in enumerate(sample_queues):
                sample_queues[qid].put("END")
            lock.release()
            print("WORK SLICER RESPONDING TO END")
            break
        if sample_nodes == "EPOCH":
            lock.acquire()
            print("PUTTING EPOCH")
            for qid,q in enumerate(sample_queues):
                sample_queues[qid].put("EPOCH")
            lock.release()
            continue
        csample = sampler.getSample(sample_nodes)
        tensorized_sample = Sample(csample)
        gpu_local_samples = []
        dummy = []
        for gpu_id in range(4):
            # gpu_local_samples.append(Gpu_Local_Sample(tensorized_sample, gpu_id))
            gpu_local_samples.append(Gpu_Local_Sample(tensorized_sample, gpu_id).serialize())
        lock.acquire()
        for qid,q in enumerate(sample_queues):
            sample_queues[qid].put(gpu_local_samples[qid])
        lock.release()
    print("Waiting for sampler process to return")
    while(True):
        time.sleep(1)
        if(sample_queues[3].qsize()==0):
            break
    time.sleep(30)
    print("SAMPLER PROCESS RETURNS")


def train(args):
    graph_name = args.graph
    dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize)
    partition_map = partition_map.type(torch.LongTensor)
    features = dg_graph.ndata["features"]
    features = features.pin_memory()
    features.share_memory_()
    cache_percentage = args.cache_per
    batch_size = args.batch_size
    no_epochs = args.num_epochs
    minibatch_size = batch_size
    fanout = args.fan_out.split(',')
    fanout = [(int(f)) for f in fanout]
    fanout = [10,10,10]
    no_worker_process = 1
    # Create main objects
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage, \
                    fanout, batch_size,  partition_map, deterministic = args.deterministic)
    storage_vector = []
    for i in range(4):
        storage_vector.append(mm.local_to_global_id[i].tolist())

    work_queue = mp.Queue(8)
    train_mask = dg_graph.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()
    if args.deterministic:
        train_nid = torch.arange(dg_graph.num_nodes())
        # When debugging at node level
        # train_nid = torch.arange(0,32)
    minibatches_per_epoch = int(len(train_nid)/minibatch_size)

    print("Training on num nodes = ",train_nid.shape)

    # global train_nid_list
    # train_nid_list= train_nid.tolist()

    work_producer_process = mp.Process(target=(work_producer), \
                  args=(work_queue, train_nid, minibatch_size, no_epochs,\
                    no_worker_process, args.deterministic))
    work_producer_process.start()

    queue_size = 8
    sample_queues = [mp.Queue(queue_size) for i in range(4)]
    lock = torch.multiprocessing.Lock()

    slice_producer_processes = []
    for proc in range(no_worker_process):
        slice_producer_process = mp.Process(target=(slice_producer), \
                      args=(graph_name, work_queue, sample_queues, lock,\
                                minibatches_per_epoch, no_epochs,storage_vector, args.deterministic))
        slice_producer_process.start()
        slice_producer_processes.append(slice_producer_process)

    procs = []
    n_gpus = 4
    labels = dg_graph.ndata["labels"]
    labels.share_memory_()
    for proc_id in range(n_gpus):
        p = mp.Process(target=(run_trainer_process), \
                      args=(proc_id, n_gpus, sample_queues[proc_id], minibatches_per_epoch \
                       , features, args, \
                       num_classes, mm.batch_in[proc_id], labels,no_worker_process, args.deterministic))
        p.start()
        procs.append(p)
    for sp in slice_producer_processes:
        sp.join()
    print("Sampler returned")
    for p in procs:
        p.join()
    print("Setup Done")

if __name__=="__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv")
    # training details
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num-workers', type=int, default=0,
       help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--fsize', type = int, default = -1, help = "use only for synthetic")
    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-per', type =float, default = .25)
    argparser.add_argument('--model-name',help="gcn|gat")
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='10,10,25')
    argparser.add_argument('--batch-size', type=int, default=(4096))
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--deterministic', default = False, action="store_true")
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    mp.set_start_method('spawn')
    train(args)
