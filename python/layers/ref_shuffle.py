from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time,datetime
from layers.shuffle_functional import *

class ShuffleRev(torch.autograd.Function):

    # from_sizes: Shapes exppected from other gpus.
    # To offsets that iwll be shuffled.
    @staticmethod
    def forward(ctx, local_t, device_id, layer_id ,from_nds,\
                to_tensor_offset):
        recv = []
        recv_g = []
        send_dict = []
        for i in range(4):
            # Do I do work allocation here ?
            if i == device_id:
                recv.append(torch.empty([0,*local_t.shape[1:]], device = device_id))
                recv_g.append(torch.empty([0,*local_t.shape[1:]], device = device_id))
                send_dict.append(None)
            else:
                # remote has the same shape as local
                recv_g.append(torch.empty((from_nds[i].shape[0], *local_t.shape[1:]) \
                    , device = device_id))
                recv.append(torch.empty((to_tensor_offset[i+1] - to_tensor_offset[i], *local_t.shape[1:]) \
                    , device = device_id))
                send_dict.append(local_t[from_nds[i]].detach())
        shuffle_functional(device_id, send_dict, recv)
        ctx.device_id = device_id
        ctx.layer_id = layer_id
        ctx.recv_g = recv_g
        ctx.back = torch.zeros(local_t.shape, device = device_id)
        ctx.from_nds = from_nds
        ctx.to_offsets =  to_tensor_offset
        # torch.cuda.current_stream().synchronize()
        cat = []
        for i in range(4):
            if i != device_id:
                cat.append( recv[i])
        remote_out = torch.cat(cat,dim = 0)
        return remote_out

    @staticmethod
    def backward(ctx, grad0):
        device_id = ctx.device_id
        recv_g = ctx.recv_g
        layer_id = ctx.layer_id
        send_grads = {}
        offset = ctx.to_offsets
        grad0 = grad0.clone()
        for i in range(4):
            send_grads[i] = grad0[offset[i]: offset[i+1]].detach()
        shuffle_functional(device_id,send_grads, recv_g)
        grads = []
        local_g = ctx.back
        for i in range(4):
            if i!= device_id:
                local_g[ctx.from_nds[i]] += recv_g[i]
        # Cross test if this local_g is getting aggregated. 

        return  local_g, None, None, None, None


class ToySingle(torch.nn.Module):

    def __init__(self,  device_id):
        super(ToySingle, self).__init__()
        self.ll = torch.nn.Linear(100,100)
        self.device_id = device_id
        self.ll.weight = torch.nn.Parameter(
            torch.ones(self.ll.weight.shape))

    def forward(self, local_input, remote_input):
        local_input = self.ll(local_input)
        remote_input = self.ll(remote_input)
        to_offsets = [0]
        from_sizes = []
        for i in range(4):
            if i == self.device_id:
                to_offsets.append(to_offsets[-1])
                from_sizes.append(None)
            else:
                to_offsets.append(to_offsets[i] + 25)
                from_sizes.append(25)
        r1,r2,r3,r4 = Shuffle.apply(remote_input, self.device_id, 0,  from_sizes,\
                    to_offsets)
        r = [r1,r2,r3,r4]
        for i in range(4):
            if i == self.device_id:
                continue
            local_input += r[i]
        return local_input


# Not  a real correctness test. Just for me to know the shuffle works
# Hence not migrating
def test_single(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = n_gpus
    pg = th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)

    model = ToySingle(proc_id).to(proc_id)
    model = DistributedDataParallel(model, device_ids = [proc_id])
    local = torch.ones((25,100),requires_grad = True, device = proc_id)
    remote = torch.ones((75,100), requires_grad = True, device = proc_id)

    out = model.forward(local, remote)
    out.sum().backward()
    print(local.grad, remote.grad)

if __name__ == "__main__":
    # Create unit test which can handle  shuffling
    procs = []
    n_gpus = 4
    print("Launch multiple gpus")
    for proc_id in range(n_gpus):
        p = mp.Process(target=(test_single),
                       args=(proc_id, n_gpus))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
