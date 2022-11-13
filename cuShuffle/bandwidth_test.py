import torch
import shuffle
import time

def test_bandwidth():
    a = torch.randint(0,1000,(1024 * 1024, (int)(1024/4)), device = 0, dtype = torch.float)
    b = torch.randint(0,1000,(1024 * 1024, (int)(1024/4)), device = 1, dtype = torch.float)
    c = torch.randint(0,1000,(1024 * 1024, (int)(1024/4)), device = 0, dtype = torch.float)
    d = torch.randint(0,1000,(1024 * 1024, (int)(1024/4)), device = 1, dtype = torch.float)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    for i in range(10):
        t1 = time.time()
        torch.cuda.set_device(0)
        e1.record()
        shuffle.copy_tensor(a,b)
        e2.record()
        e2.synchronize()
        t2 = time.time()

        # print("measured", t2-t1, e1.elapsed_time(e2)/1000)
        print("bandwidth",(a.shape[0] * a.shape[1])*4/(1024**3)/(t2-t1), "GBPs")
        # t1 = time.time()
        # e1.record()
        # b[:,:] = a[:,:]
        # e2.record()
        # t2 = time.time()
        # e2.synchronize()
        # print("py measured", t2-t1, e1.elapsed_time(e2)/1000)
        # print("py bandwidth",(a.shape[0] * a.shape[1])*4/(1024**3)/(e1.elapsed_time(e2)/1000), "GBPs")

test_bandwidth()

shuffle.test_bandwidth()
