import torch
import dgl
import time
import nvtx
from models.dist_gcn import get_sage_distributed
from models.dist_gat import get_gat_distributed
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager, GpuLocalStorage
import torch.optim as optim
from cslicer import cslicer
from data import Bipartite, Sample, Gpu_Local_Sample
import numpy as np
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os
import time
import inspect
from utils.shared_mem_manager import *
from data.serialize import *
from utils.log import *

def avg(ls):
    # assert(len(ls) > 3)
    print(ls)
    if(len(ls) <= 3):
        return sum(ls)/len(ls)
    a = max(ls[1:])
    b = min(ls[1:])
    # remove 3 as (remove first, max and min)
    return (sum(ls[1:]) - a - b)/(len(ls) - 3)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum(),len(pred )

def get_sample(proc_id, sample_queues,  sm_client, log):
    sample_id = None
    device = proc_id
    t0 = time.time()
    if proc_id == 0:
        log.log("leader tries to read meta data, qsize {}".format(sample_queues[0].qsize()))
        meta = sample_queues[0].get()
        log.log("leader reads meta data, starts sharing")
        for i in range(1,4):
            if type(meta) == tuple:
                sample_queues[i].put(meta[i])
            else:
                sample_queues[i].put(meta)
        if type(meta) == tuple:
            meta = meta[0]
        log.log("leader done sharing")
    else:
        log.log("followers tries to read")
        meta = sample_queues[proc_id].get()
        log.log("follower gets meta")
    log.log("Meta data read {}".format(meta))
    t1 = time.time()
    sample_get_time = t1-t0
    graph_move_time = 0
    if(type(meta) == type("")):
        gpu_local_sample = meta
        sample_id = meta
    else:
        (name, shape, dtype ) = meta
        dtype = np.dtype( dtype )
        log.log("trying to reconstruct data on shared memory")
        t3 = time.time()
        tensor = sm_client.read_from_shared_memory(name, shape, dtype)
        t4 = time.time()
        log.log("data reconstruction complete")
        tensor = tensor.to(device)
        print("Warning: Impromptu data reformatting is risky. ")
        # tensor = tensor.long()
        gpu_local_sample = Gpu_Local_Sample()
        device = torch.device(device)
        # Refactor this must not be moving to GPU at this point.
        construct_from_tensor_on_gpu(tensor, device, gpu_local_sample)
        gpu_local_sample.prepare()
        t5 = time.time()
        # Memory can be released now as object is constructed from tensor.
        sm_client.free_used_shared_memory(name)
        log.log("construction of sample on gpu {}".format(gpu_local_sample.randid))
        sample_id = gpu_local_sample.randid
        sample_get_time += (t4 - t3)
        graph_move_time = (t5-t4)
    return sample_id, gpu_local_sample, sample_get_time, graph_move_time



def run_trainer_process(proc_id, gpus, sample_queue, minibatches_per_epoch, features, args\
                    ,num_classes, batch_in, labels, num_sampler_workers, deterministic,in_degrees
                    , sm_filename_queue, cached_feature_size, cache_percentage):
    gpu_local_storage = GpuLocalStorage(cache_percentage, features, batch_in, cached_feature_size, proc_id)
    print("Num sampler workers ", num_sampler_workers)
    log = LogFile("Trainer", proc_id)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = gpus
    torch.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    print("SEED",torch.seed())
    sm_client = SharedMemClient(sm_filename_queue, "trainer", proc_id)
    # Use this when I need to match accuracy
    if deterministic:
        print("not setting seeds, use this to match accuracy")
    #     set_all_seeds(seed)
    if args.model == "gcn":
        model = get_sage_distributed(args.num_hidden, features, num_classes,
            proc_id, args.deterministic, args.model)
    else:
        assert(args.model == "gat")
        model = get_gat_distributed(args.num_hidden, features, num_classes,
            proc_id, args.deterministic, args.model)
    model = model.to(proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id],\
                output_device = proc_id)
                # find_unused_parameters = False
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
    labels= labels.to(proc_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    i = 0
    sample_get_epoch = []
    forward_epoch = []
    backward_epoch = []
    movement_graph_epoch = []
    movement_feat_epoch = []
    epoch_time = []
    epoch_accuracy = []
    sample_get_time = 0
    forward_time = 0
    backward_time = 0
    movement_graph = 0
    movement_feat = 0
    in_degrees = in_degrees.to(proc_id)
    fp_start = torch.cuda.Event(enable_timing=True)
    fp_end = torch.cuda.Event(enable_timing=True)
    bp_end = torch.cuda.Event(enable_timing=True)
    accuracy = 0
    torch.cuda.set_device(proc_id)
    nmb = 0
    ii = 0
    e_t1 = time.time()
    print("Features ", features.device)
    num_epochs = 0
    while(True):
        nmb += 1
        log.log("blocked at get sample")
        sample_id, gpu_local_sample, sample_get_mb, graph_move_mb = get_sample(proc_id, sample_queue,  sm_client, log)
        sample_get_time += sample_get_mb
        movement_graph += graph_move_mb
        log.log("sample recieved and processed")
        if(gpu_local_sample == "EPOCH"):
            e_t2 = time.time()
            optimizer.zero_grad()
            sample_get_epoch.append(sample_get_time)
            forward_epoch.append(forward_time)
            backward_epoch.append(backward_time)
            movement_graph_epoch.append(movement_graph)
            movement_feat_epoch.append(movement_feat)
            epoch_time.append(e_t2-e_t1)
            sample_get_time = 0
            forward_time = 0
            backward_time = 0
            movement_graph = 0
            movement_feat = 0
            e_t1 = time.time()
            num_epochs += 1
            continue
        if(gpu_local_sample == "END"):
            #print("GOT END OF FLAG")
            num_sampler_workers -= 1
            #print("got end of epoch flag")
            # Maintain num active sampler workers
            if num_sampler_workers == 0:
                break
            else:
                continue

        #assert(features.device == torch.device('cpu'))
        #gpu_local_sample.debug()
        classes = labels[gpu_local_sample.last_layer_nodes].to(torch.device(proc_id))
        # print("Last layer nodes",gpu_local_sample.last_layer_nodes)
        # with nvtx.annotate("forward",color="blue"):
        torch.cuda.set_device(proc_id)
        optimizer.zero_grad()
        #with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        m_t0 = time.time()
        input_features  = gpu_local_storage.get_input_features(gpu_local_sample.missing_node_ids)
        m_t1 = time.time()
        movement_feat += (m_t1-m_t0)
        fp_start.record()
        assert(features.device == torch.device('cpu'))
        #print("Start forward pass !")

        output = model.forward(gpu_local_sample, input_features, in_degrees)
        # continue
        if args.deterministic:
            print("Expected value", output.sum(), gpu_local_sample.debug_val)
            continue
        # torch.cuda.set_device(proc_id)
        fp_end.record()
        #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        # print(prof.key_averages(group_by_input_shape=True))
        loss = loss_fn(output,classes)/args.batch_size
        #assert(classes.shape[0] != 0)
        #print("loss",loss)
        loss.backward()
        # continue
        # print("backward complete",proc_id)
        for p in model.parameters():
            p.grad *= 4
        if deterministic:
            model.module.print_grad()
        ii = ii + 1
        bp_end.record()

        if output.shape[0] !=0:
            acc = compute_acc(output,classes)
            acc = (acc[0].item()/acc[1])
        torch.cuda.synchronize(bp_end)
        forward_time += fp_start.elapsed_time(fp_end)/1000
        # print("Forward time",fp_start.elapsed_time(fp_end)/1000 )
        # with nvtx.annotate("backward", color="red"):
        backward_time += fp_end.elapsed_time(bp_end)/1000
        if deterministic:
            model.module.print_grad()
        optimizer.step()

    print("Exiting main training loop")
    dev_id = proc_id
    # if proc_id == 1:
    #     prof.dump_stats('worker.lprof')
    if proc_id == 0:
        print("accuracy:{}".format(acc))
        print("#################",epoch_time)
        print("epoch:{}".format(avg(epoch_time)))
        print("sample_time:{}".format(avg(sample_get_epoch)))
        print("movement graph:{}".format(avg(movement_graph_epoch)))
        print("movement feature:{}".format(avg(movement_feat_epoch)))
        print("forward time:{}".format(avg(forward_epoch)))
        print("backward time:{}".format(avg(backward_epoch)))
        # print("Memory",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
    # print("Thread running")
