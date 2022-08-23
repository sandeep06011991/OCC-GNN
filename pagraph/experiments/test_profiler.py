import torch
import time

"Torch profiler events made absolutely no sense to me"
"This is some skeleton code where I explored that scenario"

def process(proc_id):
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    torch.cuda.set_device(proc_id)
    A = torch.rand(10000,10000,device = proc_id)
    B = torch.rand(10000,10000,device = proc_id)
    t1 = time.time()
    e1.record()
    C1 = A @ B
    C2 = A @ B
    C3 = A @ B
    e2.record()
    t2 = time.time()
    time.sleep(1)
    e3.record()
    with torch.autograd.profiler.profile(enabled = (proc_id == 0), use_cuda = True) as prof:
        a = torch.rand(1000000)
        with torch.autograd.profiler.record_function('mask'):
            for i in range(10):
                b = a.to(proc_id)
                b = b * 2
    t3 = time.time()
    e3.synchronize()
    print("Cuda elapsed time1",e1.elapsed_time(e2))
    print("Cuda elapsed time2",e2.elapsed_time(e3))
    print("Total measured command line time",t2 - t1)
    print(prof.key_averages())

for i in range(4):
    process(i)
