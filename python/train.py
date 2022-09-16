import torch
import dgl
import time
import nvtx
from models.factory import get_model_distributed
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager
import torch.optim as optim
from cslicer import cslicer
from data import Bipartite, Sample, Gpu_Local_Sample
import numpy as np
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os
#os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/cslicer/"
import time
import inspect
from utils.shared_mem_manager import *
from data.serialize import *

def get_sample_and_deserialize(sample_queue, sm_client, device):
    t = sample_queue.get()
    if(type(t) == type("")):
        gpu_local_sample = t
        sample_id = t
    else:
        # Unpack
        t1 = time.time()
        (name, shape, dtype ) = t
        dtype = np.dtype( dtype )
        print('a',dtype)
        tensor = sm_client.read_from_shared_memory(name, shape, dtype)
        print('b',tensor.dtype)
        tensor = tensor.to(device)
        tensor = tensor.long()
        gpu_local_sample = Gpu_Local_Sample()
        device = torch.device(device)
        # Refactor this must not be moving to GPU at this point.
        construct_from_tensor_on_gpu(tensor, device, gpu_local_sample)
        t2 = time.time()
        print("Deserialization cost", t2-t1)
        sample_id = gpu_local_sample.randid
    return str(sample_id), gpu_local_sample

def get_sample_and_sync(proc_id, sample_buffer,
        store, sample_queue, sm_filename_queue):
    sample_id = None
    if proc_id == 0 or len(sample_buffer) == 0:
        sample_id, gpu_local_sample = get_sample_and_deserialize\
                (sample_queue, sm_filename_queue, proc_id)
    if sample_id  == None:
        sample_id = list(sample_buffer.keys())[0]
        gpu_local_sample = sample_buffer[sample_id]
    # Handle race condition
    if proc_id == 0:
        store.set("leader",str(sample_id))
    torch.distributed.barrier()
    if proc_id != 0:
        try:
            leader_id = store.get("leader")
            leader_id = leader_id.decode("utf-8")
            while(sample_id != leader_id):
                print("mismatch", sample_id, leader_id)
                sample_buffer[sample_id] = gpu_local_sample
                if leader_id not in sample_buffer:
                    sample_id, gpu_local_sample = get_sample_and_deserialize(sample_queue, sm_filename_queue, proc_id)
                else:
                    sample_id = leader_id
                    gpu_local_sample = sample_buffer[leader_id]
            if sample_id in sample_buffer:
                del sample_buffer[sample_id]
        except:
            print("gpu 0 is shut down")
            gpu_local_sample = "EPOCH"
            sample_id = "EPOCH"
    return sample_id, gpu_local_sample



def run_trainer_process(proc_id, gpus, sample_queue, minibatches_per_epoch, features, args\
                    ,num_classes,batch_in, labels, num_sampler_workers, deterministic,in_degrees
                    , sm_filename_queue):
    print("Num sampler workers ", num_sampler_workers)

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = gpus
    torch.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    print("SEED",torch.seed())
    sm_client = SharedMemClient(sm_filename_queue)
    if deterministic:
        set_all_seeds(seed)
    model = get_model_distributed(args.num_hidden, features, num_classes, proc_id, args.deterministic)
    model = model.to(proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id],\
                output_device = proc_id)
                # find_unused_parameters = False
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
    labels= labels.to(proc_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    i = 0
    sample_get_time_epoch = []
    forward_time_epoch = []
    backward_time_epoch = []
    sample_move_time_epoch = []
    epoch_time = []
    in_degrees = in_degrees.to(proc_id)
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
    if proc_id == 0:
        store = dist.TCPStore("127.0.0.1", 1234, 4, True)
    else:
        store = dist.TCPStore("127.0.0.1",1234, 4, False)

    sample_buffer = {}

    # if proc_id == 10:
    #     prof = line_profiler.LineProfiler()
    #     f = prof(get_sample_and_deserialize)
    # else:
    #     f = get_sample_and_deserialize
    while(True):

        t11 = time.time()
        nmb += 1
        if proc_id == 0:
            print("minibatch",nmb)
        t111 = time.time()
        sample_id, gpu_local_sample = get_sample_and_sync(proc_id, sample_buffer,
            store, sample_queue,  sm_client)
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
            #print("GOT END OF FLAG")
            num_sampler_workers -= 1
            #print("got end of epoch flag")
            # Maintain num active sampler workers
            if num_sampler_workers == 0:
                break
            else:
                continue
        t33 = time.time()
        gpu_local_sample.prepare()
        #assert(features.device == torch.device('cpu'))
        #gpu_local_sample.debug()
        t44 = time.time()
        sample_move_time += t44 - t33
        classes = labels[gpu_local_sample.last_layer_nodes].to(torch.device(proc_id))
        # print("Last layer nodes",gpu_local_sample.last_layer_nodes)
        # with nvtx.annotate("forward",color="blue"):
        torch.cuda.set_device(proc_id)
        optimizer.zero_grad()
        #with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        fp_start.record()
        assert(features.device == torch.device('cpu'))
        #print("Start forward pass !")
        output = model.forward(gpu_local_sample,batch_in, in_degrees)
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
        #print("backward complete",proc_id)
        #if(classes.shape[0]):
        #    print("Backward not blocks when classes is zero")
        # Helpful trick to make manual calculation of gradients easy.
        # torch.sum(output).backward()

        for p in model.parameters():
            p.grad *= 4
        if deterministic:
            model.module.print_grad()
        ii = ii + 1
        bp_end.record()

        if False  and output.shape[0] !=0:
            acc = compute_acc(output,classes)
            print("accuracy {}",acc)
        torch.cuda.synchronize(bp_end)
        forward_time += fp_start.elapsed_time(fp_end)/1000
        print("Forward time",fp_start.elapsed_time(fp_end)/1000 )
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
        print("avg forward time: {}sec, device {}".format(sum(forward_time_epoch[1:])/(num_epochs - 1), dev_id))
        print(forward_time_epoch)
        print("avg backward time: {}sec, device {}".format(sum(backward_time_epoch[1:])/(num_epochs - 1), dev_id))
        print("avg move time: {}sec, device {}".format(sum(sample_move_time_epoch[1:])/(num_epochs - 1), dev_id))
        print(sample_move_time)
        print('avg epoch time: {}sec, device {}'.format(sum(epoch_time[1:])/(num_epochs - 1), dev_id))
        print(epoch_time)
        print('avg sample get time: {}sec, device {}'.format(sum(sample_get_time_epoch[1:])/(num_epochs - 1),dev_id))
        print(sample_get_time_epoch)
    # print("Memory",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
    # print("Thread running")
