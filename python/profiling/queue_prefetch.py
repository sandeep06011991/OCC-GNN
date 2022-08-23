
import torch
import torch.multiprocessing as mp
import subprocess
import time
# prefetch factor simply ensures that at the start there is something in the queue
# can easily be simulated by waiting

class A:

    def __init__(self):
        self.data = torch.rand(100000,100)
        

def producer(queue):
    data = []
    for i in range(10):
        data.append(A())
    s = time.time()
    for d in data:
        queue.put(d)
    e = time.time()
    print("Producer bandwidth time",e-s)
    while queue.qsize() != 0:
        time.sleep(1)
    print("producer exits")

def consumer(queue):
    t1 = time.time()
    for i in range(10):
        a = queue.get()
        #print(a,i)
    t2 = time.time()
    print("Consumer bandwidth time", t2- t1)


if __name__ == "__main__":
    q = mp.Queue() 
    p = mp.Process(target = producer,args = (q,))
    p.start()
    time.sleep(3)
    p = mp.Process(target = consumer, args = (q,))
    p.start()
    p.join()
