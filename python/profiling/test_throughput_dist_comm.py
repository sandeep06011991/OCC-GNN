# What is the best format for communication between processes
# Torch distributed sending which can potentially speed up using nvlink
from torch.multiprocessing import Queue
import torch as th
import torch.multiprocessing as mp
import time
import torch

def run_use_multi_queues_per_process(proc_id, n_gpus, args, queues, devices):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    input_t = torch.rand(100,100, device = proc_id)
    device_id = proc_id
    for i in range(10):
        t1 = time.time()
        for qid,q in enumerate(queues):
            if qid != device_id:
                a = input_t.detach().share_memory_()
                queues[qid][device_id].put((device_id,a))
        torch.distributed.barrier()
        for i in range(4):
            if i==device_id:
                continue
            (from_id, data) = queues[device_id][i].get()
            input_t += data.to(device_id)
        t2 = time.time()
        if device_id == 0:
            print("Total time using multiple queues per process", t2-t1)

def run_use_single_queues_per_process(proc_id, n_gpus, args, queues, devices):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    input_t = torch.rand(100,100, device = proc_id)
    device_id = proc_id
    for i in range(10):
        t1 = time.time()
        for qid,q in enumerate(queues):
            if qid != device_id:
                a = input_t.detach().share_memory_()
                queues[qid][qid].put((device_id,a))
        torch.distributed.barrier()
        for i in range(3):
            (from_id, data) = queues[device_id][device_id].get()
            input_t += data.to(device_id)
        t2 = time.time()
        if device_id == 0:
            print("Total time using single queue per process", t2-t1)


def using_dist_send(proc_id, n_gpus, args, queues, devices):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    input_t = torch.rand(100,100, device = proc_id)
    temp = [torch.rand(100,100, device = proc_id) for i in range(4)]
    device_id = proc_id
    if proc_id == 0:
        #torch.distributed.barrier(device_ids = [proc_id])
        print("Proof of useless barrier")
    for i in range(10):
        t1 = time.time()
        send_queue = []
        for to_id in range(4):
            if to_id == device_id:
                continue
            a = input_t.detach().share_memory_()
            send_queue.append(torch.distributed.isend(a,to_id,tag = device_id))
        torch.distributed.barrier()
        irecv_queue = []
        for from_id in range(4):
            if from_id == device_id:
                continue
            irecv_queue.append(torch.distributed.irecv(temp[from_id], src=from_id, tag=from_id))
        for obj in irecv_queue:
            obj.wait()
        for from_id in range(4):
            if from_id == device_id:
                continue
            input_t += temp[from_id]
        t2 = time.time()
        if device_id == 0:
            print("Total time using distributed send", t2-t1)

if __name__ == "__main__":
    n_gpus = 4
    communication_queues = [[Queue(4) for j in range(4)] for i in range(4)]
    args = ()
    devices = 4
    procs =  []
    test_functions = [run_use_multi_queues_per_process,\
            run_use_single_queues_per_process,\
            using_dist_send]
    test_functions = [using_dist_send]
    for f in test_functions:
        for proc_id in range(n_gpus):
            p = mp.Process(target=(f),
                           args=(proc_id, n_gpus, args, communication_queues, devices))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
