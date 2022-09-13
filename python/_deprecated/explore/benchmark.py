
import torch
import time
# t1 = torch.rand(100*10000).to('cuda')
# del t1
t1 = time.time()
for i in range(10000):
    t = torch.rand(i*100).to('cuda:0')
    torch.cuda.
    del t

t2 = time.time()
print("total time",t2 - t1)
