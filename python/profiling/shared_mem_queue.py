import time
import multiprocessing as mp
import threading

from multiprocessing.shared_memory import SharedMemory
import numpy as np


def producer(queue, size, proc_id, no_objects, reuse_queue, configure_O):
    creation_time = 0
    producer_time = 0
    q = []
    mg = []
    for i in range(no_objects):
        mmg = SharedMemory(name = 'id{}'.format(i),create= True, size =8 * 100000)
        a = np.ndarray((100000),dtype = int,  buffer = mmg.buf)
        b = np.ones(100000,dtype = int)
        a[:] = b[:]
        print(np.sum(a))
        q.append(i)
        mg.append(mmg)
    for i in range(no_objects):
        t0 = time.time()
        #if configure_O == "largeRandom":
        #  o = dgl.rand_graph(10**5, 10**6)
        #elif configure_O == "splitRandom":
        #  splitFactor = 4
        #  o = dgl.rand_graph(10**5, 10**6)
         # o.create_formats_()
        #  print(o.formats())
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
        o = q[i]
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
    for m in mg:
        m.close()
        m.unlink()
def consumer(queue, no_objects, proc_id, reuse_queue):
    pop_time = 0
    move_time = 0
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
        #print(obj.shape)
        t0 = time.time()
        #mmg = SharedMemory(name = 'id{}'.format(obj), size = 8 * 100000)
        #a = np.ndarray((100000),dtype = int,  buffer = mmg.buf)
        #a = np.ndarray((100000),dtype = int,  buffer = mmg.buf)
        #print(np.sum(a))
        #mmg.close()
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
    no_objects = 2
    no_procs = 1 
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
