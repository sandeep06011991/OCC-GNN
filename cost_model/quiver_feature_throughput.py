import quiver
import torch.distributed as dist
import torch 
import os 
import torch.multiprocessing as mp 

def run(rank, feature, start, end):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    torch.cuda.set_device(rank)

    dist.init_process_group('nccl', rank=rank, world_size=4)
    dist.barrier()
    feature.lazy_init_from_ipc_handle()
    #print(feature.cpu_part)
    print(feature.clique_tensor_list[0].shard_tensor_config.tensor_offset_device)

    # Test local bandwidth 
    # MB to GB
    proc_id = rank
    torch.cuda.set_device(proc_id)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    dim = feature.shape[1]
    for size in [1024 * 1024, 1024 **2 * 16, 1024 **2  * 128, 1024 ** 3]:
        idx = int(size/(dim * 4 * 4))
        idxes = torch.randint(start[proc_id], end[proc_id] , (idx,) , device = proc_id) 
        for it in range(4):
            dist.barrier()
            e1.record()
            y = feature[idxes]
            e2.record()
            e2.synchronize()
            dist.barrier()
            time = e1.elapsed_time(e2)/1000
            print(it, "Size" , size/ (1024 * 1024) ,"MB", "Local Bandwidth", time, size / (time * 1024 ** 3))
    # Test remote all goes to same peer bandwith 
    for size in [1024 * 1024, 1024 **2 * 16, 1024 **2  * 128, 1024 ** 3]:
        idx = int(size/(dim * 4 * 4))
        idxes = torch.randint(start[1], end[1] , (idx,) , device = proc_id)
        for _ in range(4):
            torch.distributed.barrier()
            e1.record()
            y = feature[idxes]
            e2.record()
            e2.synchronize()
            time = e1.elapsed_time(e2)/1000
            print("All access same peer", "Size" , size/ (1024 * 1024) ,"MB", "Local Bandwidth", time, size / (time * 1024 ** 3))
    # Test spread out bandwidth
    for size in [1024 * 1024, 1024 **2 * 16, 1024 **2  * 128, 1024 ** 3]:
        idx = int(size/(dim * 4 * 4))
        peer_id = proc_id
        idxes = torch.randint(start[(peer_id+1)%4], end[(peer_id + 1)%4] , (idx,) , device = proc_id)
        for _ in range(4):
            dist.barrier()
            e1.record()
            y = feature[idxes]
            e2.record()
            e2.synchronize()
            time = e1.elapsed_time(e2)/1000
            print("Neighbour Peer without overlap", "Size" , size/ (1024 * 1024) ,"MB", "Local Bandwidth", time, size / (time * 1024 ** 3))
    # Full Spred out
    for size in [1024 * 1024, 1024 **2 * 16, 1024 **2  * 128, 1024 ** 3]:
        idx = int(size/(dim * 4 * 4))
        idxes = torch.randint(start[0], end[3] , (idx,) , device = proc_id)
        for _ in range(4):
            dist.barrier()
            e1.record()
            y = feature[idxes]
            e2.record()
            e2.synchronize()
            time = e1.elapsed_time(e2)/1000
            print("Spreadacross all peers", "Size" , size/ (1024 * 1024) ,"MB", "Local Bandwidth", time, size / (time * 1024 ** 3))

   


if __name__=="__main__":
    cache_per = .25
    dim = 128
    nGB = int(4 * (1024 * 1024 * 1024 )/ ( dim * 4))
    feat = torch.rand(nGB, dim)
    quiver.init_p2p(device_list = [0,1,2,3])
    cache_size = int(float(cache_per) * feat.shape[0] * feat.shape[1] * 4/(1024 * 1024))
    device_cache_size = "{}M".format(cache_size)
    cache_policy = "p2p_clique_replicate"
            #for device in range(4):
            #    start = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].start)
            #    end = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].end)
            #    offsets[device] = (start,end)
    nfeat = quiver.Feature(rank=0, device_list=[0,1,2,3],
                               #device_cache_size="200M",
                               device_cache_size = device_cache_size,
                               cache_policy = cache_policy,
                               #cache_policy="device_replicate",
                               csr_topo=None)
    #feat = dg_graph.in_degrees().unflatten(0, (dg_graph.num_nodes(), 1)) * torch.ones(dg_graph.num_nodes(), 10, dtype = torch.float32)
    #print(feat.shape)
    nfeat.from_cpu_tensor(feat)
    #print(nfeat.feature_order.shape)
    #print(graph.out_degrees(nfeat.feature_order.to('cpu')))
    start = {}
    last_node_stored = {}
    assert(len(nfeat.clique_tensor_list) == 1)
    for i in range(4):
        start[i]  = nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[i].start
        last_node_stored[i] = nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[i].end
        # Temporary disable
        #for device in range(4):
        #    start = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].start)
        #    end = (nfeat.clique_tensor_list[0].shard_tensor_config.tensor_offset_device[device].end)
        #    offsets[device] = (start,end)
        #    print(device, nfeat[start:end])
    print("Not doing UNIFIED MEMORY")
    


    print('============================')
    #print(f'Final Test: {test_accs.mean():.4f} ± {test_accs.std():.4f}')
    world_size = 4
    mp.spawn(
        run,
        args=(nfeat, start, last_node_stored),
        nprocs=world_size,
        join=True
    )

