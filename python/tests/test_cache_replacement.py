from utils.utils import get_process_graph
from cslicer import cslicer
from utils.memory_manager import *
# P1This is a correctness test .
# What about a perf test

from data.part_sample import *

def get_graph(graph_name):
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    cache_percentage = .25
    fanout = [10,10,10]
    batch_size = 4096

    mm = MemoryManager( dg_graph, dg_graph.ndata['features'], dg_graph.ndata['labels'], cache_percentage, \
            fanout, batch_size, partition_map)
    local_storage = [GpuLocalStorage(cache_percentage, dg_graph.ndata['features'], mm.batch_in[i], i) for i in range(4)]

    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    gpu_map = [i.tolist() for i in mm.local_to_global_id]
    deterministic = False
    testing = False
    self_edge = False
    rounds = 1
    pull_optimization = False
    no_layers = 3
    fanout = 10
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    csample = slicer.getSample(train_nid[:4096].tolist())
    csample  = Sample(csample)
    local_sample = [Gpu_Local_Sample() for i in range(4)]


    for i in range(4):
        local_sample[i].set_from_global_sample(csample, i)
    for j in range(4):
        out_nodes = local_sample[i].out_nodes
        out_labels = dg_graph.ndata['labels'][out_nodes].to(j)
        cache_hit_from = local_sample[i].cache_hit_from
        cache_hit_to = local_sample[i].cache_hit_to
        cache_miss_from = local_sample[i].cache_miss_from
        cache_miss_to = local_sample[i].cache_miss_to
        local_storage[i].get_input_features(cache_hit_from, cache_hit_to , \
                cache_miss_from, cache_miss_to)

def baseline_pagraph():
    # Start graph server Assume this as external

    Storage(graph, node_num, nid_map, gpuid, cache_per)
    # nit field auto cache
    for cache_hit percentages from 0 to 100
    self.gpu_flag[nids] = True
    for j in range(10):
        Miss rate
        fetch_data(self, input_nodes)
    # Storage function

# get_graph("ogbn-arxiv")
