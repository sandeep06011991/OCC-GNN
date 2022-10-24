import torch
import os
device = int(os.environ['CUDA_VISIBLE_DEVICES'])

gb1 = int((1024 * 1024 * 1024)/4)


for i in range(16):
    t1 = torch.zeros((i * gb1,1),device = torch.device(device))
    print("able to allocate {}",i)



