import torch
import time
import torch.multiprocessing as mp
import dgl
import queue as py_queue
import threading


class object():
    def __init__(self, size, proc_id):
        self.data = torch.ones(size,)
        # self.data.share_memory_()
        self.size = size
        self.proc_id = proc_id

    def to(self, device):
        self.data = self.data.to(self.proc_id)
        assert(torch.sum(self.data).item() == self.size)


def producer(queue, size, proc_id, no_objects, reuse_queue, configure_O):
    creation_time = 0
    producer_time = 0
    for i in range(no_objects):
        t0 = time.time()
        if configure_O == "largeRandom":
          o = dgl.rand_graph(10**5, 10**6)
        elif configure_O == "splitRandom":
          splitFactor = 4
          o1 = dgl.rand_graph((10**5)/splitFactor, (10**6)/splitFactor)
          o = [o1.clone() for _ in range(splitFactor)]
        # if False:
        #     # if i>10:
        #     o1 = reuse_queue.get()
        #     o = torch.ones(size, out=o1)
        # else:
        #     o1 = torch.ones(10000)
        #     o = [o1.clone(), o1.clone(), o1.clone(), o1.clone()]
          # o = [[[i]] for i in range(10000)]
          # o = dgl.rand_graph(1000,100000)
          # o.share_memory_()
        # o = object(size,proc_id)
        t1 = time.time()
        queue.put(o)
        t2 = time.time()
        # if i > 100:
        creation_time += t1-t0
        producer_time += t2-t1
    print("creation time", creation_time)
    print("queue put time", producer_time)
    while(queue.qsize() != 0):
        time.sleep(2)


def consumer(queue, no_objects, proc_id, reuse_queue):
    pop_time = 0
    move_time = 0
    local_queue = py_queue.Queue(10)
    # def prefetch_func(local_queue,global_queue):
    #     while True:
    #         a = global_queue.get()
    #         local_queue.put(a)
    #         # print("put")
    # th = threading.Thread(target = prefetch_func,args = (local_queue,queue))
    # th.start()
    time.sleep(3)
    for i in range(no_objects * proc_id):
        t0 = time.time()
        obj = queue.get()
        t1 = time.time()
        # obj.clone().to(0)
        t2 = time.time()
        # if i > 400:
        pop_time += t1 - t0
        move_time += t2 - t1
        # reuse_queue.put(obj)
    print("queue pop time", pop_time)
    print("move_time", move_time)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Try all variations of queues
    # import multiprocessing
    queue = mp.Queue(30)
    reuse_queue = mp.Queue(100)
    no_objects = 20
    no_procs = 4
    size = 3500000
    pp = []
    for i in range(no_procs):
        producer_process = mp.Process(target=(producer),
                                      args=(queue, size, i, no_objects, reuse_queue, "splitRandom"))
        producer_process.start()
        pp.append(producer_process)
    consumer_process = mp.Process(target=(consumer), args=(
        queue, no_objects, no_procs, reuse_queue))
    consumer_process.start()
    consumer_process.join()

    for p in pp:
        p.join()
