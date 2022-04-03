import torch
import time
def run_experiment(warm_up):
    size = [100 * 1024 * 1024, 1024 * 1024 * 1024 ]
    A = torch.rand((1024 ,128),device = "cuda:0")
    B = torch.rand(( 128, 1024),device = "cuda:0")

    for sz in size:
        a = torch.rand(sz)
        t1 = time.time()
        for i in range(2000):
            c1 = A@B
        # a.to('cuda:0')
        t2 = time.time()
        print(sz/(1024*1024),"MB","time",t2-t1)

def run_event(warm_up):
    size = [100 * 1024 * 1024, 1024 * 1024 * 1024 ]
    A = torch.rand((1024 ,128),device = "cuda:0")
    B = torch.rand(( 128, 1024),device = "cuda:0")

    for sz in size:
        a = torch.rand(sz)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t1 = time.time()
        start.record()
        for i in range(2000):
            c1 = A@B
        end.record()
        torch.cuda.synchronize(end)
        t2 = time.time()
        print("From sync time", start.elapsed_time(end)/1000)
        # print(sz/(1024*1024),"MB",t2-t1)


run_experiment(True)
run_experiment(False)
run_experiment(False)
run_experiment(False)

run_event(True)
run_event(True)
run_event(True)
run_event(True)
run_event(True)

# Without polluting async
# 100MB .08s
# 1GB 1.32s
