import torch
from cslicer import work
import torch.multiprocessing as mp
import time
#  Proof that processes cannot communicate through shared threading module.
# Cmodule [1] = PyShell[Worker] == Trainer
def consumer_process(proc_id,n_gpus, queue, num_objects):
    for i in range(num_objects):
        a = queue.get_object(proc_id)
        t = a.t
        print(t.shape)
        print(t[:10])
        print("read sum", proc_id , i , "data", torch.sum(t))
        # return
def single_process(proc_id,n_gpus, queue, num_objects):
    for i in range(num_objects):
        for j in range(4):
            a = queue.get_object(j)
            t = a.t
            print(t.shape)
            print(t[:10])
            print("read sum", proc_id , i , "data", torch.sum(t))

import logging
logging.basicConfig(filename='example.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
class Object():

    def __init__(self, val):
        self.a = val

def proc1(proc_id, obj):
    logging.debug("proc 1")
    print(proc_id, obj.a)
    obj.a = 2
    print(proc_id, obj.a)

def proc2(proc_id, obj):
    time.sleep(4)
    logging.debug("proc 2")
    print(proc_id, obj.a)


if __name__ == "__main__":
    num_objects = 10
    num_threads = 3
    n_gpus = 4
    # queue = work(num_objects, num_threads)
    proc_id = 0
    # time.sleep(10)
    # single_process(proc_id, n_gpus, queue, num_objects * num_threads)
    pp = []
    obj = Object(10)
    p = mp.Process(target = (proc1), args = (0, obj))
    p.start()
    p1 = mp.Process(target = (proc2), args = (1, obj))
    p1.start()
    p1.join()
    # for proc_id in range(n_gpus):
    #     p = mp.Process(target=(consumer_process), \
    #                   args=(proc_id, n_gpus, queue, num_objects * num_threads)
    #                   )
    #     pp.append(p)
    #     p.start()
    #
    # for p in pp:
    #     p.join()
