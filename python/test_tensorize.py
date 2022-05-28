import torch

import random
import time
torch.tensor([1,2,3],device = 0)
t1 = time.time()
l = []
randomlist = []
singlelist = []
for j in range(100):
    randomlist = []
    for i in range(0,100000):
        n = random.randint(1,30)
        randomlist.append(n)
        singlelist.append(n)
    l.append(randomlist)
        
t1 = time.time()
for ll in l:
    t = torch.tensor(randomlist,device = 0)
t2 = time.time()
t = torch.tensor(singlelist, device = 0)
t3 = time.time()
t = torch.tensor(singlelist)
t4 = time.time()
t.to(device = 0)
t5 = time.time()
data_size = len(singlelist) * 4 /(1024 * 1024)

print("moved data", len(singlelist) * 4 / (1024 * 1024))
print("aggr. bandwidth", len(singlelist) * 4 /((t3-t2) * 1024 * 1024))
print("total time ", t2 - t1)
print("aggr. bandwidth without tensorization", data_size / (t5 - t4))
print("aggregated bandwidth", t3- t2)
print("just data movement", t5-t4)
