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


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))




def run(proc_id, n_gpus, args, devices):
    # Start up distributed training, if enabled.
    print("starting sub process", proc_id)
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    out = th.cuda.set_device(dev_id)

    model1 = ToyModel()
    model1.net1.weight = torch.nn.Parameter(torch.ones(10,10)* dev_id)
    model1 = model1.to(dev_id)
    print("pre dist sync", torch.sum(model1.net1.weight))   
    if n_gpus > 1:
        model = DistributedDataParallel(model1, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("post dist sync", torch.sum(model1.net1.weight))    

    
    for epoch in range(args.num_epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        inp_data = torch.rand(100,10).to(dev_id)
        batch_labels = torch.randint(0,5,(100,)).to(dev_id)
        batch_pred = model(inp_data)
        
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        print("epoch", epoch, "device", dev_id,"weight",torch.sum(model1.net1.weight))
        torch.distributed.barrier()
        optimizer.step()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--batch-size', type=int, default=1032)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0)
    args = argparser.parse_args()
    n_gpus =3 
    devices = [1,2,3]
    start_time = time.time()
    if n_gpus == 1:
        # assert(false)
        print("Running on single GPUs")
        run(0, n_gpus, args, devices, data)
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
