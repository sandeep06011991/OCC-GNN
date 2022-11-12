gr# What is the best format for communication between processes
# Torch distributed sending which can potentially speed up using nvlink
import torch as th
import torch.multiprocessing as mp
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

comm_map = {
    0:[1,2,3],
    1:[0,3,2],
    2:[3,0,1],
    3:[2,1,0]

        }

def using_dist_send_sync_co_ordinated(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    #GB = 1024 * 1024
    j = 0
    device_id = proc_id
    data = [torch.rand(((int)(GB / 4) ,),device = proc_id) for i in range(4)]


    num_tries = 10
    for _ in range(num_tries):
        t1 = time.time()
        for i in range(3):
            peer_id = comm_map[device_id][i]
            if(peer_id < device_id):
                torch.distributed.send(data[device_id], peer_id)
                torch.distributed.recv(data[peer_id], src = peer_id)
            else:
                torch.distributed.recv(data[peer_id], src = peer_id)
                torch.distributed.send(data[device_id], peer_id)
        torch.distributed.barrier()
        t2 = time.time()

        print("Time ", t2-t1, "GBps",  12 * 1/(t2-t1))

# Calculates bandwith of point to point dist send and recieve.
def using_dist_send_buffer_blocked(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    GB = 1024 * 1024
    j = 0
    device_id = proc_id
    data = [torch.rand(((int)(GB / 4) ,),device = proc_id) for i in range(4)]


    num_tries = 10
    for _ in range(num_tries):
        t1 = time.time()
        for i in range(4):
            if device_id == i:
                continue
            torch.distributed.send(data[device_id], i)
        for i in range(4):
            if device_id == i:
                continue
            torch.distributed.recv(data[i], src = i)
        torch.distributed.barrier()
        t2 = time.time()
        print("Time ", t2-t1, "GBps", 12 *  1/(1024 * (t2-t1)))

# Calculates bandwith of point to point dist send and recieve.
def using_dist_send_p2p(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    #GB = 1024 * 1024
    j = 0
    device_id = proc_id
    data = [torch.rand(((int)(GB / 4) ,),device = proc_id) for i in range(4)]


    num_tries = 10
    for _ in range(num_tries):
        t1 = time.time()
        if device_id == 0:
            #o1 = torch.distributed.isend(data[device_id], 1)
            #o2 = torch.distributed.isend(data[device_id], 2)
            o3 = torch.distributed.isend(data[device_id], 3)
            o1 = torch.distributed.isend(data[device_id], 1)
            o2 = torch.distributed.isend(data[device_id], 2)
            o1.wait()
            o2.wait()
            o3.wait()
        if device_id == 1:
            torch.distributed.irecv(data[0], src = 0)
        if device_id == 2 or device_id == 3:
            torch.distributed.irecv(data[0], src = 0)
        if device_id == 3:
            torch.distributed.irecv(data[0], src = 0)
        torch.distributed.barrier()
        t2 = time.time()
        if device_id == 0:
            print("Check Time ", t2-t1, "GBps",  1/(t2-t1))


def using_dist_async(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    GB =  1024 * 1024 * 1024
    j = 0
    device_id = proc_id
    data = [torch.rand(((int)(GB / 4) ,),device = proc_id) for i in range(4)]


    num_tries = 10
    for _ in range(num_tries):
        t1 = time.time()
        send = []
        recv = []
        for i in range(3):
            peer_id = comm_map[device_id][i]
            if(peer_id < device_id):
                send.append(torch.distributed.isend(data[device_id], peer_id))
                recv.append(torch.distributed.irecv(data[peer_id], src = peer_id))
            else:
                recv.append(torch.distributed.irecv(data[peer_id], src = peer_id))
                send.append(torch.distributed.isend(data[device_id], peer_id))
        torch.distributed.barrier()
        t2 = time.time()
        for s in send:
            s.wait()
        for r in recv:
            r.wait()
        torch.distributed.barrier()
        t2 = time.time()
        if device_id == 0:
            print("Time ", t2-t1, "Bandwidth", 12 * 1/(t2-t1))


def pcie_data_transfer(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    model = torch.nn.Sequential(torch.nn.Linear(128,10))
    model = model.to(proc_id)
    model = th.nn.parallel.DistributedDataParallel(model,device_ids = [proc_id])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    x = torch.rand(1000000,128)
    x = x.pin_memory()
    for i in range(2):
        y = x.to(proc_id)
        o = model(y)
        optimizer.step()
    #with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True,) as prof:
    t00 =  time.time()
    for i in range(10):
        t1 = time.time()
           # with torch.profiler.record_function("movement"):
        y = x.to(proc_id,non_blocking = True)
        y = y * 2
        print(y[0][0])
        t2 = time.time()
        print("Time per call",t2-t1,(1000 * 1000 * 128/((1024*1024)*(t2-t1))), "MBPS")
        #out = model(y)
        #out.sum().backward()
        #optimizer.step()
    print("Total time ", time.time() - t00)
    #print(prof.key_averages())

    print("all done !!")

def using_dist_overhead(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    t1 = time.time()
    torch.cuda.set_device(proc_id)
    for i in range(100):
        pass
    t2 = time.time()
    for i in range(100):
        th.distributed.barrier(device_ids = [proc_id])
    t3 = time.time()
    print("base line",t2 - t1)
    print("dist barrier", t3 - t2)


'''
Single gpu transfer total time .4 secnds total, bandwitdh 2 gbps, .04 seconds per movement
two to four gpus transfer time  .4 - .8 seconds per gpu. bandwirdh 2 gps .06 - .08 seconds per iteration
'''
if __name__ == "__main__":
    n_gpus = 4
    n_gpus = 4
    procs = []
    # assert(False)
    test_functions = [using_dist_send_p2p]
    test_functions = [using_dist_send_buffer_blocked]
    #test_functions = [using_dist_async]
    #test_functions = [using_dist_send_sync_co_ordinated]
    for f in test_functions:
        for proc_id in range(n_gpus):
            p = mp.Process(target=(f),
                           args=(proc_id, n_gpus))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
