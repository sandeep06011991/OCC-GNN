# What is the best format for communication between processes
# Torch distributed sending which can potentially speed up using nvlink
import torch as th
import torch.multiprocessing as mp
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

comm_map = {
    0:[1,2,3],
    1:[0,3,2],
    2:[3,0,1],
    3:[2,1,0]
}

comm_map = {
    0:[1,2,3,4,5,6,7],
    1:[0,3,2,5,4,7,6],
    2:[3,0,1,6,7,4,5],
    3:[2,1,0,7,6,5,4],
    4:[5,6,7,0,1,2,3],
    5:[4,7,6,1,0,3,2],
    6:[7,4,5,2,3,0,1],
    7:[6,5,4,3,2,1,0],
}


# All data over here should not have any gradients
# They are handled seperately.
def shuffle_functional(device_id, send_dict, recv_dict, num_devices):
    t1 = time.time()
    send = []
    recv = []
    for i in range(7):
        peer_id = comm_map[device_id][i]
        if peer_id >= num_devices:
            continue
        if(peer_id < device_id):
            if send_dict[peer_id].shape[0] != 0:
                send.append(torch.distributed.isend(send_dict[peer_id], peer_id))
            if recv_dict[peer_id].shape[0] != 0:
                recv.append(torch.distributed.irecv(recv_dict[peer_id], src = peer_id))
        else:
            if recv_dict[peer_id].shape[0] != 0:
                recv.append(torch.distributed.irecv(recv_dict[peer_id], src = peer_id))
            if send_dict[peer_id].shape[0] != 0:
                send.append(torch.distributed.isend(send_dict[peer_id], peer_id))
    for r in recv:
        r.wait()
    for s in send:
        s.wait()


def using_dist_send_sync_co_ordinated(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    GB = 1024
    j = 0
    device_id = proc_id
    data_send = [torch.ones(((int)(GB / 4) ,) * proc_id,device = proc_id) for i in range(4)]
    data_recv = [torch.ones(((int)(GB / 4) ,) * -1,device = proc_id) for i in range(4)]


    num_tries = 10
    for _ in range(num_tries):
        t1 = time.time()
        for i in range(3):
            peer_id = comm_map[device_id][i]
            if(peer_id < device_id):
                torch.distributed.send(data[device_id], peer_id)
                torch.distributed.recv(data[peer_id], src = peer_id)
            else:
                torch.distributed.recv(data[peer_id], src = peer_id)
                torch.distributed.send(data[device_id], peer_id)
        torch.distributed.barrier()

        t2 = time.time()

        print("Time ", t2-t1, "GBps",  12 * 1/(t2-t1))

def using_dist_async(proc_id, n_gpus):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='30099')
    world_size = n_gpus
    th.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    GB  = 1024 * 1024 * 1024
    GB =  1024
    j = 0
    device_id = proc_id
    send_data = {}
    recv_data = {}

    num_tries = 4
    for k in range(num_tries):
        for i in range(n_gpus):
            send_data[i] = torch.ones(((int)(GB / 4) ,) ,device = proc_id) *   proc_id * k
            recv_data[i] = torch.rand(((int)(GB / 4) ,),device = proc_id)
        t1 = time.time()
        shuffle_functional(device_id, send_data, recv_data, n_gpus)
        t2 = time.time()
        print("Time ", t2-t1, "Bandwidth", ((n_gpus-1) * n_gpus * 1)/(t2-t1))
        for i in range(n_gpus):
            if i!=proc_id:
                print(recv_data[i][i] == i * k)



'''
Single gpu transfer total time .4 secnds total, bandwitdh 2 gbps, .04 seconds per movement
two to four gpus transfer time  .4 - .8 seconds per gpu. bandwirdh 2 gps .06 - .08 seconds per iteration
'''
if __name__ == "__main__":
    n_gpus = 4
    n_gpus = 3
    procs = []
    # assert(False)
    test_functions = [using_dist_async]
    for f in test_functions:
        for proc_id in range(n_gpus):
            p = mp.Process(target=(f),
                           args=(proc_id, n_gpus))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
