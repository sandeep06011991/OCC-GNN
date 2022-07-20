import dgl
import torch
from dgl.nn.pytorch import SAGEConv
import dgl.function as fn
import torch.multiprocessing as mp
import time
import torch as th
from torch.nn.parallel import DistributedDataParallel

import torch.optim as optim

class ToySingle(torch.nn.Module):

    def __init__(self,inf,outf,device_id):
        super(ToySingle, self).__init__()
        self.ll = SAGEConv(inf,outf,'mean').to(device_id)

    def forward(self,g,f):
        return self.ll(g,f)

def run_single(proc_id,n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    model = ToySingle(100, 100, proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id], output_device = proc_id)
    g1 = dgl.rand_graph(1000,100000).to(torch.device(proc_id))
    f = torch.rand(1000,100,device=torch.device(proc_id))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ll = SAGEConv(100,100,'mean').to(torch.device(proc_id))
    optimizer = optim.Adam(model.parameters(), lr=.001)
    for i in range(10):
        start.record()
        out = model(g1,f)
        out.sum().backward()
        optimizer.step()
        end.record()
        optimizer.zero_grad()
        torch.cuda.synchronize(end)
        if proc_id == 0:
            print("time fixed kernel ",start.elapsed_time(end)/1000)
    print("Mark Done")

def test_multiprocess():
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    for proc_id in range(n_gpus):
        p = mp.Process(target=(run_single),
                       args=(proc_id, n_gpus ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        
test_multiprocess()
