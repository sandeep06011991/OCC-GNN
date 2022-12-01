from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time,datetime
from layers.shuffle_functional import *

class Pull(torch.autograd.Function):

    # from_sizes: Shapes exppected from other gpus.
    # To offsets that iwll be shuffled.
    @staticmethod
    def forward(ctx, local_t, device_id, layer_id ,pull_from_offsets,\
                    push_to_ids):
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
                recv.append(torch.empty((pull_from_offsets[i + 1] - pull_from_offsets[i], *local_t.shape[1:]) \
                    , device = device_id))
                recv_g.append(torch.empty(push_to_ids[i].shape[0], *local_t.shape[1:]) \
                    , device = device_id))
                send_dict.append(local_t[push_to_ids[i]].detach())
        shuffle_functional(device_id, send_dict, recv)
        ctx.device_id = device_id
        ctx.layer_id = layer_id
        ctx.recv_g = recv_g
        ctx.pull_from_offsets = pull_from_offsets
        ctx.push_to_ids = push_to_ids
        ctx.local_size = local_t.shape[0]
        ret = [local_t]
        for i in range(4):
            if i != device_id:
                ret.append(recv[i])
        return torch.cat(ret, dim = 0)        
        # torch.cuda.current_stream().synchronize()
        # return recv[0],recv[1],recv[2], recv[3]

    @staticmethod
    def backward(ctx, grad0):
        # Aggregate and merge gradients.
        send_grads = []
        pull_from_offsets = ctx.pull_from_offsets
        for i in range(4):
            if i==device:
                send_grads.append(None)
            else:
                send_grads.append(grad0[pull_from_offsets[i]:pull_from_offsets[i+1]].detach())
        # send_grads = [grad0.clone(), grad1.clone(),grad2.clone(), grad3.clone()]
        device_id = ctx.device_id
        recv_g = ctx.recv_g
        layer_id = ctx.layer_id
        shuffle_functional(device_id,send_grads, recv_g)
        grads = []
        grad0 = grad0.clone()[:ctx.local_size]
        for i in range(4):
            if i!= device_id:
                grad0[ctx.push_to_ids[i]] += recv_g[i]
        
        # torch.cuda.current_stream().synchronize()
        return  grad0, None, None, None, None, None


def pull(bipartite_graph, local_out, device_id, layer_id):
    # of Higher size.
    new_out = Pull.apply(local_out, device_id, layer_id, bipartite_graph.pull_from_offsets, bipartite_graph.push_to_ids)
    return new_out

