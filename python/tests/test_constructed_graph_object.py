import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *

def test_heterograph_construction_python():
    graph_name = "ogbn-arxiv"
    gpu_map = [[] for _ in range(4)]
    fanout = 10
    deterministic = True
    testing = False
    self_edge = False
    rounds = 1
    pull_optimization = False
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization)
    csample = slicer.getSample([i for i in range(100)])
    global_sample = Sample(csample)
    local_samples = [Gpu_Local_Sample() for i in range(4)]
    for i in range(4):
        local_samples[i].set_from_global_sample(global_sample, i)
        local_samples[i].prepare()
if __name__=="__main__":
    test_heterograph_construction_python()
