import torch
import time

a = torch.rand(100000,100)
a.pin_memory()
for i in range(10):
    t1 = time.time()
    b = a.to(0)
    t2 = time.time()
    print("PCIe",t2-t1)
a = a.to(0)

e1 = torch.cuda.Event(enable_timing = True)
e2 = torch.cuda.Event(enable_timing = True)
for i in range(10):
    t1 = time.time()
    e1.record()
    b = a.to(1)
    e2.record()
    t2 = time.time()
    e2.synchronize()
    print("NVlink",t2-t1,e1.elapsed_time(e2)/1000)


