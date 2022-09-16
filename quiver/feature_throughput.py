#(/home/q91/condaenv) q91@halfadgx:~$ cat test_quiver.py 
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import time

####################
# Import Quiver
####################
import quiver

def run(rank, world_size, quiver_feature):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    nds = torch.arange(0,1024*1024)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    for i in range(0,1024*1024,65536):
        e1.record()
        t1 = time.time()
        f = quiver_feature[nds[i:i+65536]]
        t2 = time.time()
        e2.record()
        e2.synchronize()
        t = e1.elapsed_time(e2)/1000
        print("move", t , rank, f.shape[0] * f.shape[1] * 4/((t) * (1024 * 1024 * 1024)))
        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    size =  1 * 1024 * 1024 * 1024
    nodes = 1024 * 1024
    dim = 256
    feature = torch.rand(nodes,dim)
    print("total size" , nodes * dim * 4/(1024 * 1024 * 1024), "GB")
    world_size = 4
    quiver.init_p2p([0,1,2,3])
    quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)),
            device_cache_size="256M",
            cache_policy="p2p_clique_replicate", csr_topo=None)
    quiver_feature.from_cpu_tensor(feature)

    print('Let\'s use', world_size, 'GPUs!')


    mp.spawn(
        run,
        args=(world_size, quiver_feature),
        nprocs=world_size,
        join=True
    )
