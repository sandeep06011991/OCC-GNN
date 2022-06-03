import time
import torch 
import torch.autograd.profiler as profiler

from torch.profiler import profile, record_function, ProfilerActivity

a = torch.rand(10000 * 1000)
for i in range(3):
  a.to(1)
a = a.pin_memory()
t1 =time.time()
'''
with profiler.profile( with_stack = True, use_cuda = True, profile_memory=True) as prof:
  with record_function("record_to_label"):
    t1 = time.time()
    for i in range(1000):
        for j in range(4):
            c = a.to(j)	
    t2 = time.time()
print("time",t2-t1)
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
print("Bandwidth", (10000 * 1000 * 1000 * 4 * 4)/(1024 * 1024 * 1024 * (t2-t1)))
'''
a = torch.rand(10000, 10000).to(0)
b = torch.rand(10000, 10000).to(0)

with profiler.profile( use_cuda = True, profile_memory = True) as prof:
    with record_function("record_kernel_time"):
        t1 = time.time()
        for i in range(100):
            b = a * b
        #print(b[0])    
        t2 = time.time()
    with record_function("test_kernel_overlap"):
        t3 = time.time()
        for i in range(100):
            b = a * b
        print(b[0])
        t4 = time.time()
        
print("time kernle 1", t2 -t1)
print("test kernel overlap", t4 - t3)
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))

