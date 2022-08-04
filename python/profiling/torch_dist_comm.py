import dgl
import torch
from dgl.nn.pytorch import SAGEConv
import dgl.function as fn
import torch.multiprocessing as mp
import time
import torch as th
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim as optim
from datetime import timedelta


# Profiling dist process communication and bug fixing
def test_point_to_point_recv_and_send_blocking(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus

    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id,
            )
    print(th.distributed.get_backend())
    torch.cuda.set_device(proc_id)
    for l in range(10):
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                if i == proc_id:
                    a = torch.ones(10000,100).to(proc_id) * proc_id
                    o = dist.isend(a,j,tag = proc_id)
                    o.wait()
                if j == proc_id:
                    a = torch.ones(10000,100).to(proc_id)
                    o = dist.irecv(a, i , tag = i)
                    o.wait()
                    assert(torch.all(a == i))
                torch.distributed.barrier()
                print(proc_id, i, j)

def test_point_to_point_recv_and_send_async(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus

    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id,
            )
    print(th.distributed.get_backend())
    torch.cuda.set_device(proc_id)
    for l in range(10):
        aa = []
        bb = []
        oo = []
        recieved = {}
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                if i == proc_id:
                    a = torch.ones(10000,100).to(proc_id) * proc_id
                    o = dist.isend(a,j,tag = proc_id)
                    # o.wait()
                    oo.append(o)
                if j == proc_id:
                    a = torch.ones(10000,100).to(proc_id)
                    o = dist.irecv(a, i , tag = i)
                    oo.append(o)
                    recieved[i] = a
                # torch.distributed.barrier()
                print(proc_id, i, j)
        for o in oo:
            o.wait()
        for k in recieved.keys():
            assert(torch.all(recieved[k]==k))
# Profiling dist process communication and bug fixing
# Currently blocks completely
def test_point_to_point_recv_and_send_batch_async(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id,
            )
    print(th.distributed.get_backend())
    torch.cuda.set_device(proc_id)
    for l in range(10):
        s = []
        d = []
        for i in range(4):
            if i == proc_id:
                continue
            a1 = torch.ones(10000,100).to(proc_id) * proc_id
            s.append(dist.isend(a1,i,tag = proc_id))
            print("send check", i, proc_id)
            d.append(a1)
        for i in range(4):
            if i == proc_id:
                continue
            a2 = torch.ones(10000,100).to(proc_id) * proc_id
            s.append(dist.irecv(a2,i,tag = i))
            print("recv check", i , proc_id)
            d.append(a2)
        print("reached", proc_id)
        for ss in s:
            ss.wait()
        # for dd in d:
        #     del dd
        torch.distributed.barrier()
        print("batch ok!")




def test_multiprocess():
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    for proc_id in range(n_gpus):
        p = mp.Process(target=(test_point_to_point_recv_and_send_batch_async),
                       args=(proc_id, n_gpus ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

# test_tensor_movement()
# test_multiprocess()
# test_singleprocess()
test_multiprocess()
