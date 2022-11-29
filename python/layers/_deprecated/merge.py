from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time,datetime
from layers.shuffle_functional import *


class Merge(torch.autograd.Function):

    # from_sizes: Shapes exppected from other gpus.
    # To offsets that iwll be shuffled.
    @staticmethod
    def forward(ctx, local_t, merge_tensors0, merge_tensors1, merge_tensors2, merge_tensors3,\
                device_id, from_ids, local_stream, remote_stream):
        merge_tensors = [merge_tensors0, merge_tensors1, merge_tensors2, merge_tensors3]
        local_stream.wait_stream(torch.cuda.current_stream())
        remote_stream.wait_stream(torch.cuda.current_stream())
        ctx.from_ids = from_ids
        ctx.device_id = device_id
        ctx.local_stream = local_stream
        ctx.remote_stream = remote_stream

        for i in range(4):
            if i != device_id:
                merge_tensors[i].record_stream(torch.cuda.current_stream())
                local_t[from_ids[i]] += merge_tensors[i]
        return local_t

    @staticmethod
    def backward(ctx, grad):
        device_id = ctx.device_id
        from_ids = ctx.from_ids
        device_id = ctx.device_id
        local_stream = ctx.local_stream
        remote_stream = ctx.remote_stream
        local_stream.wait_stream(torch.cuda.current_stream())
        remote_stream.wait_stream(torch.cuda.current_stream())
        merge_grads = []
        for i in range(4):
            if i!= device_id:
                merge_grads.append(grad[from_ids[i]])
            else:
                merge_grads.append(None)

        return  grad, merge_grads[0], merge_grads[1], merge_grads[2], merge_grads[3], None, None, None, None
