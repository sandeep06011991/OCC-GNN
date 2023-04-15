# How to use cslicer in pythonic way.
import time
import torch
from cuslicer import cuslicer
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
num_layers = 1
p_map = np.fromfile("{}/{}/partition_map_opt_4.bin".format(DATA_DIR,graphname),dtype = np.intc)
p_map = torch.from_numpy(p_map)
training_nodes = p_map.shape[0]
training_nodes = [i for i in range(training_nodes)]
# s1 = csl1.getSample(in_nodes)
# storage_map_full = [[i for i in range(169343)] for i in range(4)]
# csl2 = cslicer(graphname, storage_map_full, 10, True, False)
# s2 = csl2.getSample(in_nodes)
storage_map_part = [torch.where(p_map == i)[0].tolist() for i in range(4)]
#CUSlicer(const std::string &name,
#      std::vector<std::vector<long>> gpu_map,
#      vector<int> fanout,
#       bool deterministic, bool testing,
#          bool self_edge, int rounds, bool pull_optimization,
#            int num_layers, int num_gpus, int current_gpu){
print("check 1")
csl3 = cuslicer(graphname, storage_map_part,
        [-1], False , False, False, 4, False, num_layers, num_gpus,0)
# deterministic = True
# testing = False
# self_edge = False
# rounds = 4
# pull_optimization = False
# num_gpus = 4
# fanout = [10]
# csl1 = cslicer(graphname, storage_map_part, fanout[0],\
#             deterministic, testing , self_edge, rounds, \
#             pull_optimization, num_layers, num_gpus)
# print("Ask for Sample")
# batch_size = 4
# i = 1
# t= 0
# s_time = time.time()
# num_gpus = 4
# in_nodes = training_nodes[i:i+batch_size]
# s3= csl3.getSample(in_nodes)
# # s4= csl1.getSample(in_nodes)
# tensorized_sample1 = Sample(s3)
# # tensorized_sample2 = Sample(s4)
# obj = Gpu_Local_Sample()
# obj.set_from_global_sample(tensorized_sample1,0)
# obj.prepare()
#
# print(obj.layers[0].graph_local)
# N = obj.layers[0].graph_local.nodes('_U').shape[0]
# obj.layers[0].indptr_L = torch.tensor([0,1,6], dtype = torch.int64)
# obj.layers[0].indices_L = torch.tensor([7, 6, 2, 5, 4, 3],  dtype = torch.int64)

# obj.layers[0].indices_L = torch.sort(obj.layers[0].indices_L)[0]
# print("Sorted",obj.layers[0].indices_L  )
# import dgl
# import torch
# import dgl.function as fn
# from dgl import heterograph_index
# from dgl.utils  import Index
# in_nodes = obj.layers[0].num_in_nodes_local
# num_out_nodes = obj.layers[0].num_out_local
# metagraph_index_local = heterograph_index.\
#     create_metagraph_index(['_U','_V_local'],[('_U','_E','_V_local')])
# obj.layers[0].indices_L.unbind()
#
# hg_local = heterograph_index.create_unitgraph_from_csr(\
#             2,  in_nodes , num_out_nodes, obj.layers[0].indptr_L.to(2),
#                 obj.layers[0].indices_L.to(2),
#                 torch.arange(obj.layers[0].indices_L.shape[0],device = 0).to(2),\
#                  "csc", transpose = True)
# graph_local = heterograph_index.create_heterograph_from_relations( \
#         metagraph_index_local[0], [hg_local], Index([in_nodes ,num_out_nodes]))
# graph_local = dgl.DGLHeteroGraph(graph_local,['_U','_V_local'],['_E'])
# print(obj.layers[0].indptr_L, obj.layers[0].indices_L)
# graph_local.nodes['_U'].data['in'] = torch.rand((in_nodes,3),requires_grad = True, device = 2)
# print(graph_local.nodes['_U'].data['in'] )
# graph_local.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
# torch.cuda.synchronize()
# print(graph_local.nodes['_V_local'].data['out'])
# print(graph_local.nodes['_V_local'].data['out'].sum().backward())
# print("All ok")

# obj.layers[0].graph_local.nodes['_U'].data['in'] = torch.rand(N,39, device = 0)
# print(obj.layers[0].graph_local.nodes['_U'].data['in'], "cHECK")
# while(i < len(training_nodes)):
#     in_nodes = training_nodes[i:i+batch_size]
#     s3= csl3.getSample(in_nodes)
#     s4= csl1.getSample(in_nodes)
#     tensorized_sample = Sample(s3)
#
#     assert(False)
#     tensorized_sample.debug()
#     sample_id = tensorized_sample.randid
#     proc_id = 0
#     for gpu_id in range(num_gpus):
#         obj = Gpu_Local_Sample()
#         obj.set_from_global_sample(tensorized_sample,gpu_id)
#         print("Temporary fix serializing on cpu")
#         data = serialize_to_tensor(obj, torch.device(proc_id))
#         data = data.to('cpu').numpy()
#         print(data)
#         print(np.sum(data))
#
#     break
#     i = i + batch_size
# e_time = time.time()
# print("Total time", e_time - s_time)
# print("Done")
