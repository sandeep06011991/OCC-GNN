import torch
import time
# for i in range(4)
i = 0
a = torch.rand(10000,10000,device = i)
b = torch.rand(10000,10000, device = i)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
ls = [None for i in range(10)]
for i in range(10):
    t1 = time.time()
    # start.record()
    ls[i] = a@b
    # ls.append(c)
    # end.record()
    t2 = time.time()
    # torch.cuda.synchronize(end)
    print("time",t2-t1)
    print("timer time",start.elapsed_time(end)/1000)

# Check alternate distributed model
import os
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
# initialize the process group
dist.init_process_group("gloo", rank=rank, world_size=len(device_ids))
