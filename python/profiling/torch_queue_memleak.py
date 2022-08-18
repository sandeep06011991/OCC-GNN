
import torch
import torch.multiprocessing as mp
import random
import time
import gc

class A:

    def __init__(self, worker_id):
        self.id = random.randint(0,10000)
        self.t = torch.rand(100,100)
        print("created",self.id, "by", worker_id)
    def __del__(self):
        print("destroyed",self.id)
def producer(queue, worker_id):
    # gc.disable()
    ls = []
    for i in range(10):
        o = A(worker_id)
        # ls.append(o)
        queue.put(o)
    while not queue.empty():
        time.sleep(1)



def consumer(queue, num_workers):
    # gc.disable()
    for j in range(10 * num_workers):
        # time.sleep(1)
        b = queue.get()
        print("recieved",b.id)

if __name__ == "__main__":
    q = mp.Queue(10)
    # b = A()
    num_workers = 6
    for i in range(num_workers):
        p = mp.Process(target = producer, args= (q,i))
        p.start()
    p = mp.Process(target = consumer, args= (q, num_workers))
    p.start()
