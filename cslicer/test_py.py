# How to use cslicer in pythonic way.

import torch
from cslicer import cslicer
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
# csl1 = cslicer(graphname, storage_map_empty, 10, True, False)
# import numpy as np
DATA_DIR = "/data/sandeep"
num_gpus = 4
num_layers = 3
p_map = np.fromfile("{}/{}/partition_map_opt_4.bin".format(DATA_DIR,graphname),dtype = np.intc)
p_map = torch.from_numpy(p_map)

in_nodes = [i for i in range(10000)]
# s1 = csl1.getSample(in_nodes)
# storage_map_full = [[i for i in range(169343)] for i in range(4)]
# csl2 = cslicer(graphname, storage_map_full, 10, True, False)
# s2 = csl2.getSample(in_nodes)
storage_map_part = [torch.where(p_map == i)[0].tolist() for i in range(4)]
# (const std::string &name,
# std::vector<std::vector<long>> gpu_map,
# int fanout,
# bool deterministic, bool testing,
#   bool self_edge, int rounds, bool pull_optimization,
#    int num_layers, int num_gpus)
csl3 = cslicer(graphname, storage_map_part,
        10, True, False, True, 4, False, num_layers, num_gpus)
s3= csl3.getSample(in_nodes)

# a = get_total_comm(s1)
# b = get_total_comm(s2)
c = get_total_comm(s3)

print("Nil", a, "Full ", b, "Part", c)
