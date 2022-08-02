from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
import torch as th
from torch.nn.parallel import DistributedDataParallel
import time,datetime
class Shuffle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_t, queues, device_id, to_dict, from_dict,layer_id):
        temp = []
        # assert(input_t.requires_grad)
        temp_g = []
        t1 = time.time()
        data = 0
        for i in range(4):
            if i==device_id:
                temp.append(None)
                temp_g.append(None)
            else:
                data += from_dict[i].shape[0] + to_dict[i].shape[0]
                temp.append(torch.empty((from_dict[i].shape[0], input_t.shape[1]) \
                    , device = device_id))
                temp_g.append(torch.empty((to_dict[i].shape[0], input_t.shape[1]) \
                    , device = device_id))
        for qid in range(4):
            if qid != device_id:
                to_id = qid
                a = input_t[to_dict[qid]].detach()
                if not a.is_shared():
                    a.detach().share_memory_()
                print("Forward pass send", torch.sum(a) )
                torch.distributed.isend(a,to_id,tag = device_id)
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
            print("Forward pass recieve", torch.sum(temp[from_id]) )
            # print("device",device_id,"recieved",temp[from_id],"put into ",from_dict[from_id],"from",from_id)
            input_t[from_dict[from_id]] += temp[from_id]
        ctx.queues = queues
        ctx.device_id = device_id
        ctx.from_dict = from_dict
        ctx.to_dict = to_dict
        ctx.temp_g = temp_g
        ctx.grad_required = input_t.requires_grad
        ctx.layer_id = layer_id
        t2 = time.time()
        # print("data movement",data/input_t.shape[0], "device", input_t.device)
        # print("shuffle forward",t2-t1,"layer",layer_id,"device",input_t.device)
        return input_t

    @staticmethod
    def backward(ctx, grad_output):
        device_id = ctx.device_id
        proc_id = device_id
        send_queue = []
        dummy = torch.tensor([10],device = device_id)
        dummy_g = [torch.tensor([10],device = device_id) for i in range(4)]
        for from_id in range(4):
            if device_id == from_id:
                continue
            send_queue.append(torch.distributed.isend(dummy,from_id,tag = device_id))
        recv_queue = []
        for to_id in range(4):
            if device_id == to_id:
                continue
            recv_queue.append(torch.distributed.irecv(dummy_g[to_id], src = to_id, tag = to_id))
        for o in recv_queue:
            o.wait()
        for i in range(10):
            if proc_id == 0:
                a = torch.rand(100,100,dtype = torch.float32).to(0) * 10
                print("forward pass blocking sending sum", torch.sum(a[:,10]))
                fp = torch.distributed.send(a[:,10].clone() , 1, tag = 1)
                # fp.wait()
            if proc_id == 1:
                b = torch.rand(100,10,dtype = torch.float32).to(1) * 2
                print("forward pass before", torch.sum(b))
                o = torch.distributed.recv(b, src = 0, tag = 1)
                # o.wait()
                print("recieved BLOCKING",torch.sum(b))
                # print("recieved XXXXXXXXXXXXXX ",torch.sum(b), o.is_completed())
        print("start back pass")
        queues = ctx.queues
        from_dict = ctx.from_dict
        to_dict = ctx.to_dict
        temp = ctx.temp_g
        layer_id = ctx.layer_id
        torch.distributed.barrier()
        t1 = time.time()
        irecv_queue = []
        t3 = time.time()

        out = grad_output.clone()
        # for obj in irecv_queue:
        print("waiting for others", device_id)
        #     print("Check completion pre ", obj.is_completed())
        # print("starting sending", grad_output)
        send_queue = []
        for from_id in range(4):
            if from_id != device_id:
                a = grad_output[from_dict[from_id]].detach()
                if layer_id == 2:
                    print("sending XXXXXXXX", from_id, a.shape, torch.sum(a))
                if not a.is_shared():
                    a.detach().share_memory_()
                send_queue.append(torch.distributed.isend(a, from_id, tag =  device_id))

        for to_id in range(4):
            # print("recv async", device_id, to_id)
            if to_id != device_id:
                temp[to_id] = torch.rand(temp[to_id].shape).to(device_id)
                if device_id == 2:
                    print("intermediate !!!!!", torch.sum(temp[to_id]))
                irecv_queue.append(torch.distributed.irecv(temp[to_id], src=to_id, tag= device_id))
        # print("recv async", device_id, to_id, irecv_queue)

        # out = grad_output
        for obj in irecv_queue:
            obj.wait()
            while(not obj.is_completed()):
                print("Async wait")
                obj.wait()
                time.sleep(1)
            print("Check completion", obj.is_completed(), device_id)
            assert(obj.is_completed())
        for obj in send_queue:
            obj.wait()
        torch.distributed.barrier()
        for to_id in range(4):
            if to_id == device_id:
                continue
            print("recieved slice !!!!!!!!!!!!",to_id, temp[to_id].shape, torch.sum(temp[to_id]))
            out[to_dict[to_id]] +=  temp[to_id]
        t2 = time.time()
        print(device_id,"returns")
        # if device_id == 2:
        #     print("Flowing gradient back", out)
        # if out.device == torch.device(0):
        #     # print("part 1", t2-t1)
        #     print("shuffle backward", t2-t1)
        # print("shuffle backward",t2-t1,"layer",layer_id,"device",grad_output.device)
        return out,None,None, None, None, None


class ToySingle(torch.nn.Module):

    def __init__(self, queues, device_id):
        super(ToySingle,self).__init__()
        self.ll = torch.nn.Linear(100,100)
        self.queues = queues
        self.device_id = device_id

    def forward(self, input, from_id, to_id):
        a = self.ll(input)
        b = Shuffle.apply(a, self.queues, self.device_id, to_id, from_id,0)
        return b

def test_single(proc_id, n_gpus, queues):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    pg = th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    from_id = {}
    to_id = {}
    for i in range(4):
        if i!=proc_id:
            from_id[i] = torch.tensor(range(100)).to(proc_id)
            to_id[i] = torch.tensor(range(100)).to(proc_id)
    model = ToySingle(queues, proc_id).to(proc_id)
    model = DistributedDataParallel(model, device_ids = [proc_id])
    X_t = torch.rand((1000,100), device = proc_id)
    for i in range(10):
        t1 = time.time()
        out = model.forward(X_t,from_id, to_id)
        t2 = time.time()
        out.sum().backward()
        t3 = time.time()
        print("Forward pass", proc_id, "time", t2-t1)
        print("Back pass", proc_id, "time", t3-t2)
    print("Forward pass working")

if __name__ == "__main__":
    # Create unit test which can handle  shuffling
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    queues = [Queue() for i in range(n_gpus)]
    for proc_id in range(n_gpus):
        p = mp.Process(target=(test_single),
                       args=(proc_id, n_gpus,queues))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
