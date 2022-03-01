import torch
import time

def slice_percentage(per,device):
    a = torch.rand(1000000,device = device)
    assert(per<1)
    t1 = time.time()
    for i in range(10):
        b = a[torch.where(a<per)]
    t2 = time.time()
    print(device,per,t2-t1)

for device in [torch.device("cpu"),torch.device("cuda:0")]:
    for per in range(0,10,1):
        slice_percentage(per/10,device)
