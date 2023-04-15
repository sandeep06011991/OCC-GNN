
import time
import torch
from cuslicer import cuslicer
import numpy as np
import nvtx
import torch
import dgl
import time
import nvtx
from models.dist_gcn import get_sage_distributed
from models.dist_gat import get_gat_distributed
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager, GpuLocalStorage
import torch.optim as optim
from data import Bipartite, Sample, Gpu_Local_Sample
import numpy as np
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os
import time
import inspect
from utils import utils
from utils.utils import *
from cu_shared import *
from data.serialize import *
import logging
from test_accuracy import *
from layers.shuffle_functional import *


# graphname = "reordered-papers100M"
# number_of_epochs = 1
# minibatch_size =4096
# num_nodes = 169343

# // Get this from data
storage_map_empty = [[],[],[],[]]
graphnames = ["ogbn-arxiv","ogbn-products"]
graphname = "ogbn-products"
# graphname = "reorder-papers100M"
# csl1 = cslicer(graphname, storage_map_empty, 10, True, False)
# import numpy as np
DATA_DIR = "/home/ubuntu/data"
num_gpus = 4
num_layers = 3
p_map = np.fromfile("{}/{}/partition_map_opt_4.bin".format(DATA_DIR,graphname),dtype = np.intc)
p_map = torch.from_numpy(p_map)
training_nodes = p_map.shape[0]
training_nodes = [i for i in range(training_nodes)]
# s1 = csl1.getSample(in_nodes)
# storage_map_full = [[i for i in range(169343)] for i in range(4)]
# csl2 = cslicer(graphname, storage_map_full, 10, True, False)
# s2 = csl2.getSample(in_nodes)
#storage_map_part = [torch.where(p_map == i)[0].tolist() for i in range(4)]
storage_map_part = [[] for i in range(4)]

# const std::string &name,
# std::vector<std::vector<long>> gpu_map,
# vector<int> fanout,
# bool deterministic, bool testing,
#   bool self_edge, int rounds, bool pull_optimization,
#     int num_layers, int num_gpus, int current_gpu
#print("check 1")
num_hidden = 128
features = torch.rand(p_map.shape[0], 128)
num_classes = 40
proc_id = 0
deterministic = False
gpus = 4
num_layers = 3
skip_shuffle = False
models = []
for proc_id in range(4):
    torch.cuda.set_device(proc_id)
    model = "gcn"
    model = get_sage_distributed(num_hidden, features, num_classes,
            proc_id, deterministic, model, gpus,  num_layers, skip_shuffle)
    model = model.to(proc_id)
    models.append(model)

csl3 = cuslicer(graphname, storage_map_part,
        [20,20,20], False , False, True, 4, False, num_layers, num_gpus,0)
print("Ask for Sample")
batch_size = 4096
i = 0
t= 0
s_time = time.time()
num_gpus = 4
while(i < len(training_nodes)):
    in_nodes = training_nodes[i:i+batch_size]
    t1 = time.time()
    with nvtx.annotate("Sample",color = 'red'):
        csample = csl3.getSample(in_nodes)
        tensorized_sample = Sample(csample)
        sample_id = tensorized_sample.randid
        sample_id = tensorized_sample.randid
        local_samples = []
        for gpu_id in range(4):
            print("gpu_id working")
            obj = Gpu_Local_Sample()
            obj.set_from_global_sample(tensorized_sample,gpu_id)
            torch.cuda.set_device(gpu_id)
            data = serialize_to_tensor(obj, torch.device(0), num_gpus = gpus)
            data = data.to(gpu_id)
            gpu_local_sample = Gpu_Local_Sample()
            device = torch.device(gpu_id)
                #print(tensor.shape, "RECOEVECD", tensor.dtype, torch.sum(tensor))
            
            construct_from_tensor_on_gpu(data, device, gpu_local_sample, num_gpus = gpus)
            #construct_from_tensor_on_gpu(tensor, device, gpu_local_sample, num_gpus = gpus)
            gpu_local_sample.prepare()
            local_samples.append(gpu_local_sample)
        
        for gpu_id, model in enumerate(models):
            torch.cuda.set_device(gpu_id)
            gpu_local_sample = local_samples[gpu_id]
            input_features  = torch.rand(gpu_local_sample.cache_hit_from.shape[0] + gpu_local_sample.cache_miss_from.shape[0], features.shape[1],\
                                device = torch.device(gpu_id))
            print(input_features.shape[0])
            models[gpu_id].layers[0].get_statistics(gpu_local_sample.layers[0],input_features, 0)
        print("Predicted Minibatch Time")            
        # Local Aggregation 
        # Measure Cross Communication
        # Further Time 

    t2 = time.time()

    # break
    i = i + batch_size
e_time = time.time()
print("Total time", e_time - s_time)
print("Done")

