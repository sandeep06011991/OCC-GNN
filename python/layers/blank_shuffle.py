from torch.multiprocessing import Queue
import torch
from torch import multiprocessing as mp
import torch as th
from torch.nn.parallel import DistributedDataParallel
import time,datetime
import torch.cuda.nvtx  as nvtx
# FixMe: Currently blocking. Test with NVLink and asynchronous send/recv
class Shuffle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_t):
        print("touch forward")

        return input_t
    
    @staticmethod
    def backward(ctx, grad):
        print("touch backward")
        return grad

a = torch.ones(3,3,requires_grad = True)
b = Shuffle.apply(a)

torch.sum(b[0:0,:]).backward()

