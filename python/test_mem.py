from utils.memory_manager import *
import torch



import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel





def run(proc_id, n_gpus, args, devices):
    # Start up distributed training, if enabled.
    print("starting sub process", proc_id)
    torch.cuda.set_device(proc_id)
    batch_in = torch.rand(1024*1024,256)
    cache_percentage = 1
    cl = GpuLocalStorage(cache_percentage, batch_in,  batch_in, 0, proc_id)
    hit = torch.randint(0,1024 * 1024, (1024,)).to(proc_id)
    missing = torch.tensor([], dtype = torch.long, device = proc_id)
    for i in range(10):
        t1 = time.time()
        a = cl.get_input_features(hit,missing)
        # a = cl.get_input_features(missing,hit)
        t2 = time.time()
        bandwidth = (a.shape[0] * a.shape[1] * 4)/(1024 * 1024 * 1024 * (t2-t1))
        print("Time", t2-t1, "Bandwidth",  bandwidth  )

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    args = argparser.parse_args()
    n_gpus =4
    devices = [0,1,2,3]
    start_time = time.time()
    if n_gpus == 1:
        # assert(false)
        print("Running on single GPUs")
        run(0, n_gpus, args, devices)
    else:
        procs = []
        print("Launch multiple gpus")
        for proc_id in range(n_gpus):
            p = mp.Process(target=(run),
                           args=(proc_id, n_gpus, args, devices))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    end_time = time.time()
    print("Total time across all processes", end_time - start_time)
