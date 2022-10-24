# How to use cslicer in pythonic way.

import torch
from cslicer import cslicer

def get_total_skew(s):
    skew = {}
    edge = []
    for id,l in enumerate(s.layers):
        if id == 2:
            for l_id,bp in enumerate(l):
                edge.append(bp.indices.shape[0])
    # print(edge)
    return (max(edge) - min(edge))/(sum(edge)/4)

# graphname = "reordered-papers100M"
# number_of_epochs = 1
# minibatch_size =4096
# num_nodes = 169343
from utils.utils import get_process_graph
# // Get this from data
import random
storage_map_empty = [[],[],[],[]]
graphnames = ["ogbn-arxiv","ogbn-products"]
graphname = "reorder-papers100M"
dg_graph, p_map, num_classes = get_process_graph(graphname, -1)

# in_nodes = [i for i in range(10000)]
# s1 = csl1.getSample(in_nodes)
# storage_map_full = [[i for i in range(169343)] for i in range(4)]
# csl2 = cslicer(graphname, storage_map_full, 10, True, False)
# s2 = csl2.getSample(in_nodes)

storage_map_part = [torch.where(p_map == i)[0].tolist() for i in range(4)]
csl3 = cslicer(graphname, storage_map_part, 20, False, False)
training_nodes = torch.where(dg_graph.ndata['train_mask'])[0].tolist()
num_nodes = len(training_nodes)
no_epochs = 3
batch_size = 4096
num_nodes = len(training_nodes)
ss = []
for epoch in range(no_epochs):
    i = 0
    random.shuffle(training_nodes)
    while(i < num_nodes):
        in_nodes = (training_nodes[i:i+batch_size])
        i = i + batch_size
        s3= csl3.getSample(in_nodes)
        ss.append(get_total_skew(s3))
print("total skew", graphname, sum(ss)/len(ss))
