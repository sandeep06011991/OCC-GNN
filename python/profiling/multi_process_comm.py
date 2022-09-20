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


class Shuffle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_t, proc_id):
        if proc_id == 0:
            a = torch.rand(100,100).to(0) * 10
            print("forward pass sending sum", torch.sum(a[:,10]))
            fp = dist.isend(torch.tensor([], device = torch.device(proc_id)) , 1, tag = 1)
            fp.wait()
        if proc_id == 1:
            b = torch.zeros(100,10).to(1) * 2
            print("forward pass before", torch.sum(b))
            o = dist.irecv(torch.tensor([], device = torch.device(proc_id), dtype = torch.float32), src = 0, tag = 0)
            o.wait()
            print("recieved ",torch.sum(b))
        ctx.proc_id = proc_id
        return input_t

    @staticmethod
    def backward(ctx, grad_output):
        proc_id = ctx.proc_id
        if proc_id == 0:
            a = torch.ones(100,100).to(0) * 10
            print("backpass sending", torch.sum(a[:,10]))
            dist.isend(a[:,10].clone() , 1, tag = 1)
        if proc_id == 1:
            b = torch.ones(100,10).to(1) * 2
            o = dist.irecv(b, src = 0, tag = 0)
            o.wait()
            if(torch.sum(b) != 1000):
                print("Actual Sum",torch.sum(b))
            assert(torch.sum(b) == 1000)
            print("back pass recieved ",torch.sum(b))
        return grad_output, None

# to profile dist process vs single process with replicate
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

import datetime
#p2p bandwidth reached is 2 Gbps. 
def dist_p2p_bandwidth(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus

    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id,
            )
    a = torch.ones(1000 * 1000, 128, device = proc_id)
    b = torch.rand(1000 * 1000, 128, device = proc_id)
    for i in range(5):
        if proc_id == 0:
            t1 = time.time()
            a[0][0] = i
            dist.send(a,1,tag = 0)
            t2 = time.time()
            print("time send",t2 -t1)
        if proc_id == 1:
            time.sleep(4)
            t1 = time.time()
            dist.recv(b,0,tag = 0)
            print(b[0][0].item(),i)
            #assert(b[0][0].item() == i)
            t2 = time.time()
            print("time recv",t2 - t1, (1000 * 1000 * 128 * 4/((t2-t1)*1024 * 1024)))
    torch.distributed.barrier()  


def dist_shuffle_bandwidth(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    #a = torch.rand(1000 * 1000, 128, device = proc_id)
    #r = [torch.rand(1000 * 1000, 128, device = proc_id) for i in range(4)]
    t1 = time.time()
    for i in range(5):
        a = torch.rand(1000 * 1000, 128, device = proc_id)
        r = [torch.rand(1000 * 1000, 128, device = proc_id) for i in range(4)]
        t1 = time.time()
        for send in range(4):
            for recv in range(4):
<<<<<<< HEAD
                if not (send == 0 and recv == 1):
                    continue                    
=======
                if not(send == 0 and recv == 1):
                    continue

>>>>>>> 1e571b7... temp changes
                if send==recv:
                    continue
                if proc_id == send:
                    dist.send(a,recv,tag = send)
                if proc_id == recv:
                    dist.recv(r[send],send,tag = send)
                    print(r[send][0][0]) 
        t2 = time.time()
        torch.distributed.barrier(device_ids = [proc_id])
        print("time recv",t2 - t1,"MBps", (1000 * 1000 * 128 * 4 /((t2-t1)*1024 * 1024)))
        del a
        del r
    torch.distributed.barrier()


# Profiling dist process communication and bug fixing
def multiprocess_with_comm_using_dist(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus

    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id,
            )
    print(th.distributed.get_backend())
    torch.cuda.set_device(proc_id)
    for i in range(5):
        print("loop")
        a = [torch.ones(100,10000).to(proc_id)* proc_id for _ in range(4)]
        b = [torch.ones(100,10000).to(proc_id) * proc_id for _ in range(4)]
        r = []
        print("start shuffle")
        t1 = time.time()
        for j in range(4):
            if j != proc_id:
                r.append(dist.isend(a[j],j,tag = proc_id))
            print("send",proc_id, j)
        for j in range(4):
            if j != proc_id:
                r.append(dist.irecv(b[j],src = j,tag = j))
        t2 = time.time()
        for i,j in enumerate(b):
            assert(j[0][0] ==i)
        t2 = time.time()    
        print("end shuffle")
        #for rr in r:
        #    rr.wait()
        t3 = time.time()
        time.sleep(2)
        print("Send time", t2-t1, "wait time",t3-t2)
        torch.distributed.barrier(device_ids = [proc_id])    
        print("success!")
        print("loop")
        continue
        a = torch.ones(100,10000).to(proc_id)* proc_id
        b = torch.ones(100,10000).to(proc_id) * proc_id
        r = []
        for j in range(4):
            if j != proc_id:
                r.append(dist.isend(a,j,tag = proc_id))
        for j in range(4):
            if j != proc_id:
                r.append(dist.irecv(b,src = j,tag = j))
        for rr in r:
            rr.wait()
        print("success!")
        continue
            #
            # o = dist.irecv(b, 0, tag = 0)
            # print("attempting to send",proc_id)
            # s = dist.isend(a, 0, tag = 1)
            # print(o.is_completed())
            # print(s.is_completed())
            # s.wait()
            # o.wait()
            # print("recieved",o.is_completed(),torch.sum(b))


            # o.wait()
        if proc_id == 0:
            a = torch.ones(100,1000).to(0)* 10
            b = torch.ones(100,1000).to(0) * 2
            print("attemtping to recv",proc_id)
            s = dist.isend(a, 1, tag = 0)
            print("attemtping to send",proc_id)
            o = dist.irecv(b, 1, tag = 1)
            print(o.is_completed())
            print(s.is_completed())
            s.wait()
            o.wait()
            print("sent",o.is_completed(),torch.sum(b))

        print("enter barrier1", proc_id)
        torch.distributed.barrier()
        # print("exit barrier1",proc_id)

    # a = torch.nn.parameter.Parameter(torch.tensor([1,2.9])).to(proc_id)
    # f =Shuffle.apply(a,proc_id)
    # for i in range(10):
    #     f.sum().backward()

# Profiling techniques for gradient exchange
def multiprocess_with_comm_using_store(proc_id, n_gpus):
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

# profiling single process replicate vs dist multi process
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
        p = mp.Process(target=(dist_shuffle_bandwidth),
                #multiprocess_with_comm_using_dist),
                #dist_p2p_bandwidth),
                       args=(proc_id, n_gpus ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

from torch.multiprocessing import Queue
# profile torch queue
def master(q,e):
    t = torch.ones(10,10, requires_grad = True).to(0)
    t = t.share_memory_()
    print(t)
    q.put(t.detach())
    e.wait()
    print(t.grad)
    print("Master puts q")

# profile torch queue
def slave(q,e):
    t = q.get()
    (t @ t).sum().backward()
    e.set()
    # print(t)
    print("Slaves get t",t)
    # del t
    # print("slave deletes t")

# profile torch queue
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

# test_tensor_movement()
# test_multiprocess()
# test_singleprocess()
test_multiprocess()
