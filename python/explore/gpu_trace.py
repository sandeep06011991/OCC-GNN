
import torch
import time
import nvtx
import threading
from torch.nn.parallel import parallel_apply
W = []
X = []
for i in range(4):
    W.append(torch.nn.Linear(1000,1000).to(torch.device(i)))
    X.append(torch.rand(10000,1000,device = i))
singleThreaded = False
if singleThreaded:
    for i in range(4):
        # with nvtx.annotate("loop", color="red"):
        t00 = time.time()
        W[i](X[i])
        t11 = time.time()
        print("inner time",t11 - t00)
else:
    parallel_apply(W,X)
t1 = time.time()
for i in range(10):
    if singleThreaded:
        for i in range(4):
            # with nvtx.annotate("loop", color="red"):
            t00 = time.time()
            out = W[i](X[i])
            t11 = time.time()
            print("inner time",t11 - t00)
    else:
        out = parallel_apply(W,X)
        # def func(i,W,X):
        #     t00 = time.time()
        #     W(X)
        #     t11 = time.time()
        #     print("inner time",i,t11 - t00)
        # threads = []
        # for i in range(4):
        #     threads.append(threading.Thread(target=func, args=(i,W[i],X[i])))
        #     threads[i].start()
        # for i in range(4):
        #     threads[i].join()
print(out)
t2 = time.time()
print("total time ", t2-t1)
