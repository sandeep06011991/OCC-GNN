import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph


def sampler_baseline_latency(graph_name):
    # graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10,10,10])
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        train_nid,
        sampler,
        device='cpu',
        batch_size= 4096,
        shuffle=True,
        drop_last=True,
        num_workers=0 )
    average_sample = []
    for i in range(3):
        t1 = time.time()
        for _ in dataloader:
            pass
        t2 = time.time()
        average_sample.append(t2-t1)
    average_sample = sum(average_sample[1:])/(len(average_sample)-1)
    return average_sample

def groot_baseline_latency(graph_name):
    # graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    gpu_map = [[] for _ in range(4)]
    fanout = 10
    deterministic = False
    testing = False
    self_edge = False
    batch_size= 4096
    rounds = 1
    pull_optimization = False
    no_layers = 3
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    average_sample = []
    for i in range(3):
        j = 0
        t1 = time.time()
        while(j < train_nid.shape[0]):
            csample = slicer.getSample(train_nid[j:j + batch_size].tolist())
            j = j + batch_size
        t2 = time.time()
        average_sample.append(t2-t1)
        # print("GRoot Sampler",t2 - t1)
    average_sample = sum(average_sample[1:])/(len(average_sample)-1)
    return average_sample

def test_sampling_overhead():
    graph = "ogbn-products"
    t1 = groot_baseline_latency(graph)
    t2 = sampler_baseline_latency(graph)
    assert(t1 < t2 * 2)

if __name__=="__main__":
    sampler_baseline_latency()
    groot_baseline_latency()
