import torch
import torch.multiprocessing as mp
import random
import time
import multiprocessing as nmp

def producer(queues, lock,wid, consensus ):

    for i in range(100):
        with lock:
            #print("has lock",wid)
            #t = random.randint(0,100000)
            t = (1000 * wid) + i
            consensus.put(t)
            for j in range(4):
                queues[j].put(t)
            #print("put",t)
            time.sleep(.00001)
        #time.sleep(.001)
    for i in range(4):
        queues[i].close()
def consumer(no_workers, queues):
    to_read = no_workers * 100
    for r in range(to_read):
        v1 = queues[0].get()
        l =  []
        l.append(v1)
        error = False
        for i in range(1,4):
            v2 = queues[i].get()
            l.append(v2)
            if(v1 != v2):
                error = True
                #print("found error")
        if error:
           print("popped",l)

if __name__ == "__main__":
    no_workers = 8
    mp.set_start_method('spawn')
    queues = [mp.Queue(3) for i in range(4)]
    consensus = mp.Queue(3)
    lock = mp.Lock()
    lock = nmp.Lock()
    for i in range(no_workers):
        p = mp.Process(target = producer, args = (queues, lock, i, consensus))
        p.start()

    c = mp.Process(target = consumer, args = (no_workers, queues, consensus))
    c.start()
    c.join()
