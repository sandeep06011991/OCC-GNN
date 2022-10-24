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

def run(rank, world_size, quiver_feature, devices ):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12310'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    device = devices[rank]
    torch.cuda.set_device(device)
    nds = torch.arange(0,1024*1024)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    print(quiver_feature.clique_tensor_list)

    for k in range(0,5):
        for i in range(0,1024*1024,1024 * 256):
            e1.record()
            t1 = time.time()
            f = quiver_feature[nds[i:i+(1024 * 256)]]
        #local_nodes = (i/(1024 * 256))
            t2 = time.time()
            e2.record()
            e2.synchronize()
            t = e1.elapsed_time(e2)/1000
        #print("CPU time", t2-t1, "CUDA time", t)
            print("move", t , device, "bandwidth",f.shape[0] * f.shape[1] * 4/((t) * (1024 * 1024 * 1024)))
            dist.barrier()


    dist.destroy_process_group()


if __name__ == '__main__':
    size =  1 * 1024 * 1024 * 1024
    nodes = 1024 * 1024
    dim = 256
    feature = torch.rand(nodes,dim)
    print("total size" , nodes * dim * 4/(1024 * 1024 * 1024), "GB")
    world_size =4
    quiver.init_p2p([0,1,2,3])
    quiver_feature = quiver.Feature(rank=1, device_list=[0,1,2,3],
            device_cache_size="10M",
            cache_policy="p2p_clique_replicate", csr_topo=None)
    quiver_feature.from_cpu_tensor(feature)
    print(quiver_feature.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[0].start)
    print(quiver_feature.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[1].end)

    print(quiver_feature.clique_tensor_list)
    print('Let\'s use', world_size, 'GPUs!')

    devices = [0,1,2,3]
    mp.spawn(
        run,
        args=(world_size, quiver_feature, devices),
        nprocs=world_size,
        join=True
        )
