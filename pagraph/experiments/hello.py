import torch
import torch.multiprocessing as mp
def func(proc_id):
    print("world")
    import sys
    sys.stdout.flush()

if __name__ == "__main__":
    mp.spawn(func, args=(), nprocs=4, join=True)
