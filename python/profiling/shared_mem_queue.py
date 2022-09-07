import time
import multiprocessing as mp
import threading
from multiprocessing.shared_memory import SharedMemory
import numpy as np


# Fix ME: 
def producer(queue, size, proc_id, no_objects, reuse_queue, configure_O):
    creation_time = 0
    queue_time = 0
    q = []
    mg = []
    for i in range(no_objects):
        t0 = time.time()
        mmg = SharedMemory(name = 'id{}'.format(i),create= True, size =8 * 100000)
        a = np.ndarray((100000),dtype = int,  buffer = mmg.buf)
        b = np.ones(100000,dtype = int)
        a[:] = b[:]
        t1 = time.time()
        creation_time  = time.time()
        print(np.sum(a))
        q.append(i)
        mg.append(mmg)
    for i in range(no_objects):
        t1 = time.time()
        queue.put(i)
        t2 = time.time()
        queput_time += t2-t1
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
    time.sleep(3)
    for i in range(no_objects * proc_id):
        t0 = time.time()
        obj = queue.get()
        t0 = time.time()
        mmg = SharedMemory(name = 'id{}'.format(obj), size = 8 * 100000)
        a = np.ndarray((100000),dtype = int,  buffer = mmg.buf)
        t1 = time.time()
        obj.clone().to(0)
        t2 = time.time()
        # if i > 400:
        pop_time += t1 - t0
        move_time += t2 - t1
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
