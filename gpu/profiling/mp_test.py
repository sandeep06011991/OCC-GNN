import torch
import time

from multiprocessing import Process
# nsys profile   --trace-fork-before-exec true   python3 mp_test.py


def run_experiment(device):
    for i in range(4):
        size = [100 * 1024 * 1024, 1024 * 1024 * 1024 ]
        A = torch.rand((1024 ,128),device = device)
        B = torch.rand(( 128, 1024),device = device)
        for sz in size:
            a = torch.rand(sz)
            t1 = time.time()
            for i in range(2000):
                c1 = A@B
            a.to(device)
            t2 = time.time()
            print(sz/(1024*1024),"MB",t2-t1)

pp = []
for i in range(4):
    p = Process(target=run_experiment, args=(i,))
    p.start()
    pp.append(p)

for p in pp:
    p.join()
