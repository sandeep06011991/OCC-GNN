

import torch
import multiprocessing as mp
import time

def func(device):
    print("Size", 10000 * 10000 * 8 / (1032 * 1032))
    t = torch.rand(10000,10000 * 8)
    t1_ = time.time()
    for i in range(100):
        t1 = time.time()
        t.to(device)
        t2 = time.time()
        print("expeceted time", (t2 - t1) * 100)
        print("bandwidth", (10000 * 10000 * 8)/(1032 * 1032 * 1032 * (t2-t1)))
    t2_ = time.time()
    print("Total time in function is ", t2_-t1_)

if __name__=="__main__":
    n_gpus =4
    t1 = time.time()
    procs = []
    for proc_id in range(n_gpus):
        p = mp.Process(target=func,\
                           args=(proc_id,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    t2 = time.time()
    print("total time", t2- t1)
