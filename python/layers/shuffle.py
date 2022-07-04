from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
import torch as th
from torch.nn.parallel import DistributedDataParallel

class Shuffle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_t, queues, device_id, to_dict, from_dict):
        for qid,q in enumerate(queues):
            if qid != device_id:
                a = input_t[to_dict[qid]].detach().share_memory_()
                q.put((device_id,a))
        torch.distributed.barrier()
        for i in range(3):
            (from_id, data) = queues[device_id].get()
            input_t[from_dict[from_id]] += data.to(device_id)
        ctx.queues = queues
        ctx.device_id = device_id
        ctx.from_dict = from_dict
        ctx.to_dict = to_dict
        return input_t

    @staticmethod
    def backward(ctx, grad_output):
        queues = ctx.queues
        device_id = ctx.device_id
        from_dict = ctx.from_dict
        to_dict = ctx.to_dict
        # print(grad_output.shape)
        for qid,q in enumerate(queues):
            if qid != device_id:
                a = grad_output[from_dict[qid]].detach().share_memory_()
                q.put((device_id,a))
        torch.distributed.barrier()
        out = grad_output.clone()
        for i in range(3):
            to_id, remote = queues[device_id].get()
            out[to_dict[to_id]] += remote.to(device_id)
        return out,None,None, None, None


class ToySingle(torch.nn.Module):

    def __init__(self, queues, device_id):
        super(ToySingle,self).__init__()
        self.ll = torch.nn.Linear(100,100)
        self.queues = queues
        self.device_id = device_id

    def forward(self, input):
        a = self.ll(input)
        from_id = {}
        to_id = {}
        for i in range(4):
            if i!=self.device_id:
                from_id[i] = torch.tensor(range(10)).to(self.device_id)
                to_id[i] = torch.tensor(range(10)).to(self.device_id)
        b = Shuffle.apply(a, self.queues, self.device_id, to_id, from_id)
        return b

def test_single(proc_id, n_gpus, queues):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    model = ToySingle(queues, proc_id).to(proc_id)
    model = DistributedDataParallel(model, device_ids = [proc_id])
    out = model.forward(torch.rand((100,100), device = proc_id))
    out.sum().backward()
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
