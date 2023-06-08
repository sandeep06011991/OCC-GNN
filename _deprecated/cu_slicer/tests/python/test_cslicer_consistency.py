# How to use cslicer in pythonic way.
import time
import torch
from cslicer import cslicer
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

import numpy as np
def get_total_comm(s):
    skew = {}
    edge = []
    for id,l in enumerate(s.layers):
        for l_id,bp in enumerate(l):
            edge.append(bp.indices.shape[0])
    return max(edge) - min(edge)/(sum(edge))/4

# graphname = "reordered-papers100M"
# number_of_epochs = 1
# minibatch_size =4096
# num_nodes = 169343

# // Get this from data
storage_map_empty = [[],[],[],[]]
graphnames = ["ogbn-arxiv","ogbn-products"]
graphname = "ogbn-arxiv"
# graphname = "ogbn-products"
# csl1 = cslicer(graphname, storage_map_empty, 10, True, False)
# import numpy as np
DATA_DIR = "/data/sandeep"
num_gpus = 4
num_layers = 2
p_map = np.fromfile("{}/{}/partition_map_opt_4.bin".format(DATA_DIR,graphname),dtype = np.intc)
p_map = torch.from_numpy(p_map)
training_nodes = p_map.shape[0]
training_nodes = [i for i in range(training_nodes)]
# s1 = csl1.getSample(in_nodes)
# storage_map_full = [[i for i in range(169343)] for i in range(4)]
# csl2 = cslicer(graphname, storage_map_full, 10, True, False)
# s2 = csl2.getSample(in_nodes)
storage_map_part = [torch.where(p_map == i)[0].tolist() for i in range(4)]

# const std::string &name,
# std::vector<std::vector<long>> gpu_map,
# vector<int> fanout,
# bool deterministic, bool testing,
#   bool self_edge, int rounds, bool pull_optimization,
#     int num_layers, int num_gpus, int current_gpu
print("check 1")

csl3 = cslicer(graphname, storage_map_part,
        10, True , False, False, 4, False, num_layers, num_gpus)
print("Ask for Sample")
batch_size = 2
i = 1
t= 0
s_time = time.time()
num_gpus = 4
while(i < len(training_nodes)):
    in_nodes = training_nodes[i:i+batch_size]
    s3= csl3.getSample(in_nodes)
    tensorized_sample = Sample(s3)
    tensorized_sample.debug()
    sample_id = tensorized_sample.randid
    proc_id = torch.device('cpu')
    for gpu_id in range(num_gpus):
        obj = Gpu_Local_Sample()
        obj.set_from_global_sample(tensorized_sample,gpu_id)
        print("Temporary fix serializing on cpu")
        data = serialize_to_tensor(obj, torch.device(proc_id))
        data = data.to('cpu').numpy()
        print(data)
        print(np.sum(data))
    break
    i = i + batch_size
e_time = time.time()
print("Total time", e_time - s_time)
print("Done")
