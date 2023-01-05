import torch
import torch.multiprocessing as mp
import time
import statistics
def process0(queue):
    t = []
    for i in range(10):
        s = time.time()
        queue.put(tuple(["ea" for _ in range(100)]))
        e = time.time()
        t.append(e-s)
    while not queue.empty():
        time.sleep(1)    
    print("Put",statistics.mean(t), statistics.variance(t))

def process1(queue, n):
    t = []
    for i in range(10 * n):
        s = time.time()
        queue.get()
        e = time.time()
        if(e-s > .1):
            print("kjlkrejalkrjalkrjlajrlea @@@@@@@@@@@@@")
        t.append(e-s)
    print("GET", statistics.mean(t), statistics.variance(t))

def run_throughput_chain():
    q = mp.Queue(64)
    np = 32
    pp = []
    for _ in range(np):
        p1 = mp.Process(target = process0, args = (q,))
        p1.start()
        pp.append(p1)
    time.sleep(2)
    p2 = mp.Process(target = process1, args = (q,np))
    p2.start()
    for p1 in pp:
        p1.join()
    p2.join()

if __name__=="__main__":
    mp.set_start_method('spawn')
    run_throughput_chain()
