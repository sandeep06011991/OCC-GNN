# What is the best format for communication between processes
# Torch distributed sending which can potentially speed up using nvlink
from torch.multiprocessing import Queue
import torch as th
import torch.multiprocessing as mp
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Calculates bandwith of point to point dist send and recieve.
def using_dist_send(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    MB = 1024 * 1024
    j = 0
    device_id = proc_id
    data = torch.rand(MB * (10 **j),device = proc_id)
    print("starting",torch.sum(data))
    for i in range(4):
        # for j in range(4):
        if True:

            data = data * j
            if proc_id == 0:
                t1 = time.time()
                torch.distributed.send(data,1,tag = proc_id)
                print("send sum",torch.sum(data))
                t2 = time.time()
                print("Send time and bandwidth",t2-t1, ((10 ** j)/(1000))/(t2-t1),"data MB",(10**j))
            if proc_id == 1:
                t1 = time.time()
                torch.distributed.recv(data,0,tag = 0)
                print("recieved sum", torch.sum(data))
                t2 = time.time()
                print("recv time and bandwidth",t2-t1, ((10 ** j)/(1000))/(t2-t1),"data MB",(10**j))
            print("current torch ",torch.sum(data))
        torch.distributed.barrier()


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
    a = 0
    # a += 0.0015239039659500123
    # a += 0.0013262720108032226
    # a += 0.001590880036354065
    # a += 0.0014962879419326781
    # a += 0.0009935680031776428
    # a += 0.0010408639907836914
    # a += 0.0009470720291137695
    # a += 0.0009839360117912292
    # a += 0.0009311040043830872
    # a += 0.0009600639939308166
    # a += 0.0009233599901199341
    # a += 0.0009565119743347168
    # print("total ", a)
    # assert(False)
    test_functions = [pcie_data_transfer]
    for f in test_functions:
        for proc_id in range(n_gpus):
            p = mp.Process(target=(f),
                           args=(proc_id, n_gpus))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
