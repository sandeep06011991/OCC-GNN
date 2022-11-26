from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time,datetime
from layers.shuffle_functional import *

class Shuffle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, remote_t, device_id, from_sizes, to_offset, layer_id):
        recv = []
        recv_g = []
        send_dict = []
        for i in range(4):
            # Do I do work allocation here ?
            if i == device_id:
                recv.append(torch.empty([0,*remote_t.shape[1:]], device = device_id))
                recv_g.append(torch.empty([0,*remote_t.shape[1:]], device = device_id))
                send_dict.append(None)
            else:
                # remote has the same shape as local
                recv.append(torch.empty((from_sizes[i]) \
                    , device = device_id))
                recv_g.append(torch.empty((to_offset[i+1] - to_offset[i], *remote_t.shape[1:]) \
                    , device = device_id))
                send_dict.append(remote_t[to_offset[i+1] - to_offset[i], :].detach())
        shuffle_functional(device_id, send_dict, recv)
        ctx.device_id = device_id
        ctx.layer_id = layer_id
        # ctx.from_sizes = from_sizes
        # ctx.to_dict = to_dict
        ctx.recv_g = recv_g

        return recv[0],recv[1],recv[2], recv[3]

    @staticmethod
    def backward(ctx, grad0, grad1, grad2, grad3):
        send_grads = [grad0.clone(), grad1.clone(),grad2.clone(), grad3.clone()]
        device_id = ctx.device_id
        # from_dict = ctx.from_dict
        # to_dict = ctx.to_dict
        recv_g = ctx.recv_g
        layer_id = ctx.layer_id
        shuffle_functional(device_id,send_grads, recv_g)
        remote_g = torch.cat(recv_g, dim = 0)
        return  remote_g, None, None, None, None


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
                from_sizes.append(torch.Size([25,100]))
        # b = Shuffle.apply(remote_input, self.device_id, from_sizes, to_offsets, 0)
        # for i in range(4):
        #     if i != self.device_id:
        #         local_input += b[i]
        return local_input

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
