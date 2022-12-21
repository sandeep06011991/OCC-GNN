import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph


def sampler_baseline_latency(graph_name, batch_size):
    # graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    sampler = dgl.dataloading.MultiLayerNeighborSampler([20,20,20])
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        train_nid,
        sampler,
        device='cpu',
        batch_size= int(batch_size/4),
        shuffle=True,
        drop_last=True,
        num_workers=0 )
    average_sample = []
    for i in range(2):
        t1 = time.time()
        for _ in dataloader:
            
            pass
        t2 = time.time()
        average_sample.append(t2-t1)
    average_sample = sum(average_sample[1:])/(len(average_sample)-1)
    return average_sample

def groot_baseline_latency(graph_name, batch_size):
    # graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    gpu_map = [[] for _ in range(4)]
    fanout = 20
    deterministic = False
    testing = False
    self_edge = False
    
    rounds = 1
    pull_optimization = False
    no_layers = 3
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    average_sample = []
    for i in range(2):
        j = 0
        t1 = time.time()
        while(j < train_nid.shape[0]):
            print(j)
            csample = slicer.getSample(train_nid[j:j + batch_size].tolist())
            j = j + batch_size
        t2 = time.time()
        average_sample.append(t2-t1)
        # print("GRoot Sampler",t2 - t1)
    average_sample = sum(average_sample[1:])/(len(average_sample)-1)
    return average_sample

def test_sampling_overhead():
    # Slicing should be atmost 2x the cost of sampling
    # As the sample is read through twice for further processing.
    agraphs = ["ogbn-arxiv"]
    batches = [1024,4096,4096 *4]
    agraphs = [ "amazon"]
    for graph in agraphs:
        for batch_size in batches:
            t1 = groot_baseline_latency(graph, batch_size)
            t2 = sampler_baseline_latency(graph, batch_size)
            with open('sampler_perf', 'a') as fp:
                fp.write("{}|{}|{}|{}\n".format(graph,batch_size,t2, t1))
    #assert(t1 < t2 * 2)

if __name__=="__main__":
    test_sampling_overhead()
    #sampler_baseline_latency()
    #groot_baseline_latency()
