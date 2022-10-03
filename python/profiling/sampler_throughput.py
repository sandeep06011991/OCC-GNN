# Measures how was cslicer works
import time
import sys
import torch
import torch.multiprocessing as mp
sys.path.append('/home/spolisetty/OCC-GNN/python/')
from cslicer import cslicer
from utils.utils import get_process_graph
from data.serialize import *
from utils.shared_mem_manager import *
from data.bipartite import *
from data.part_sample import *
from utils.memory_manager import MemoryManager
from utils.shared_mem_manager import *

def run_single_worker_process(graph_name):
    proc_id = 0
    device = 0
    sm_filename_queue = mp.Queue(NUM_BUCKETS)
    sm_manager = SharedMemManager(sm_filename_queue)
    sm_client = SharedMemClient(sm_filename_queue, "trainer", proc_id)
    dg_graph,partition_map,num_classes = get_process_graph("ogbn-arxiv", -1)
    partition_map = partition_map.type(torch.LongTensor)
    cache_percentage = .1
    batch_size = 1024
    fanout = [10,10,10]
    features = dg_graph.ndata["features"]
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage, \
                    fanout, batch_size,  partition_map, deterministic = False)
                     # args.deterministic)
    storage_vector = []
    for i in range(4):
        storage_vector.append(mm.local_to_global_id[i].tolist())
    # storage_vector = [[],[],[],[]]
    deterministic = False
    train_mask = dg_graph.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()
    print("Training nodes", train_nid.shape[0])
    sampler = cslicer(graph_name,storage_vector,10, deterministic)
    train_nid = train_nid.tolist()
    device = 0
    for j in range(6):
        list = []
        i = 0
        get_sample_time = 0
        python_time = 0
        serialize_to_object = 0
        deserialize_to_object = 0
        while i < len(train_nid):
            sample_nodes = train_nid[i:i+batch_size]
            i = i + batch_size
            t1 = time.time()
            csample = sampler.getSample(sample_nodes)
            t2 = time.time()
            tensorized_sample = Sample(csample)
            t3 = time.time()
            gpu_local_samples = []
            for gpu_id in range(4):
                # gpu_local_samples.append(Gpu_Local_Sample(tensorized_sample, gpu_id))
                obj = Gpu_Local_Sample()
                obj.set_from_global_sample(tensorized_sample,gpu_id)
                data = serialize_to_tensor(obj)
                data = data.numpy()
                name = sm_client.write_to_shared_memory(data)
                ref = ((name, data.shape, data.dtype.name))
                gpu_local_samples.append(ref)
            name, shape, dtype = gpu_local_samples[0]
            t4 = time.time()
            tensor = sm_client.read_from_shared_memory(name, shape, dtype)
            tensor = tensor.to(device)
            # tensor = tensor.long()
            gpu_local_sample = Gpu_Local_Sample()
            device = torch.device(device)
            # Refactor this must not be moving to GPU at this point.

            construct_from_tensor_on_gpu(tensor, device, gpu_local_sample)
            gpu_local_sample.prepare()
            t5 = time.time()
            for j in gpu_local_samples:
                sm_client.free_used_shared_memory(j[0])
            get_sample_time += (t2-t1)
            python_time += (t3-t2)
            serialize_to_object += (t4-t3)
            deserialize_to_object += (t5-t4)
        print("epoch get sample time", get_sample_time )
        print("tensorize_time", python_time)
        print("serialize_to_object", serialize_to_object)
        print("deserialize_to_object", deserialize_to_object)
    del sm_manager
    pass
run_single_worker_process('ogbn-arxiv')
# epoch get sample time 0.5861666202545166
# tensorize_time 1.7831141948699951
# serialize_to_object 0.8369870185852051
# deserialize_to_object 0.5380048751831055

print("done")
def run():
    pass
