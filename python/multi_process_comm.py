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

class ToySingle(torch.nn.Module):

    def __init__(self,inf,outf,device_id):
        super(ToySingle, self).__init__()
        # self.ll = SAGEConv(inf,outf,'mean').to(device_id)
        self.ll = torch.nn.Linear(100,100).to(device_id)

    def forward(self,input):
        return self.ll(input)

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


def multiprocess_with_comm(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus

    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    model = ToySingle(100, 100, proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id], output_device = proc_id)
    print("Master stores attempt")
    server_store = dist.TCPStore("127.0.0.1", 8000 + proc_id, 4, True, timedelta(seconds=1))
    print("Master stores")
    client_stores = []
    for i in range(4):
        if i==proc_id:
            client_stores.append(None)
        else:
            print(8000 + proc_id)
            client_stores.append(dist.TCPStore("127.0.0.1", 8000 + proc_id , 2, False))
    print("Created all stores")
    torch.distributed.barrier()
    input = torch.ones(100,100, device = torch.device(proc_id))
    for i in range(10):
        out = model(input)
        for j in range(4):
            if j != proc_id:
                temp = out.to(j)
                client_stores[j].set(proc_id,temp)
        torch.distributed.barrier()
        for j in range(4):
            out += server_store.get(j)
            server_store.delete_key(j)
        torch.distributed.barrier()
        print("All ok v1 communication")

def base_singleprocess():
    ll = torch.nn.Linear(100*2, 100)
    g1 = dgl.rand_graph(1000,100000)
    g2 = [dgl.heterograph({('_U','_E','_V'): g1.edges()},\
            {'_U':g1.num_nodes(), '_V':g1.num_nodes()}, device = torch.device(i)) for i in range(4)]
    f_in = [torch.rand(1000,100, device = torch.device(i)) for i in range(4)]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lls = torch.nn.parallel.replicate(ll.to(0),[0,1,2,3])

    for i in range(10):
        # torch.cuda.set_device(0)
        # start.record()
        t1 = time.time()
        for i in range(4):
            g2[i].nodes['_U'].data['in'] = f_in[i]
            g2[i].update_all(fn.copy_u('in', 'm'), fn.mean('m', 'out'))
            out = g2[i].nodes['_V'].data['out']
            out[100:] += f_in[i][100:]
            out1 = torch.cat([out,out], dim = 1)
            out2 = lls[i](out1)
        # torch.cuda.set_device(0)
        # end.record()
        t2 = time.time()
        print("Time {}",t2-t1)
#       end.record()
        # torch.cuda.synchronize(end)
        # print("time function",start.elapsed_time(end)/1000)
# print("All Done")

def test_multiprocess():
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    for proc_id in range(n_gpus):
        p = mp.Process(target=(multiprocess_with_comm),
                       args=(proc_id, n_gpus ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

from torch.multiprocessing import Queue
def master(q,e):
    t = torch.ones(10,10, requires_grad = True).to(0)

    t = t.share_memory_()
    print(t)
    q.put(t.detach())
    e.wait()
    # del t
    # import time
    # time.sleep(10)
    print(t.grad)
    print("Master puts q")

def slave(q,e):
    t = q.get()
    (t @ t).sum().backward()
    e.set()
    # print(t)
    print("Slaves get t",t)
    # del t
    # print("slave deletes t")

def test_tensor_movement():
    q = Queue()
    e = torch.multiprocessing.Event()
    p1 = mp.Process(target= master, args = (q,e))
    p1.start()
    import time
    # time.sleep(2)
    p2 = mp.Process(target = slave ,args = (q,e))
    p2.start()
    p1.join()
    p2.join()
test_tensor_movement()
# test_multiprocess()
# test_singleprocess()
