# How to use cslicer in pythonic way.
import time
import torch
from cuslicer import cuslicer
import numpy as np
import nvtx
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
training_nodes = [i for i in range(int(training_nodes/4))]
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
# (const std::string &name,
#       std::vector<std::vector<NDTYPE>> gpu_map,
#       vector<int> fanout,
#        bool deterministic, bool testing,
#           bool self_edge, int rounds, bool pull_optimization,
#             int num_layers, int num_gpus, int current_gpu, bool random){
csl3 = cuslicer(graphname, storage_map_part,
        [20,20,20], False , False, True, 4, False, num_layers, num_gpus,0,False)
print("Ask for Sample")
batch_size = 4096
i = 0
t= 0
s_time = time.time()

while(i < len(training_nodes)):
    in_nodes = training_nodes[i:i+batch_size]
    t1 = time.time()
    with nvtx.annotate("Sample",color = 'red'):
        s3= csl3.getSample(in_nodes, False)
    t2 = time.time()
    print("minibatch time", t2-t1)
    # break
    i = i + batch_size
e_time = time.time()
print("Total time", e_time - s_time)
print("Done")
