import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph


def sampler_baseline_latency(graph_name, batch_size):
    # graph_name = "ogbn-arxiv"
    # Modified to GPU
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1,4)
    dg_graph = dg_graph.to('cuda:0')
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    sampler = dgl.dataloading.MultiLayerNeighborSampler([20,20,20])
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        train_nid,
        sampler,
        device='cuda',
        batch_size= int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=0 )
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    average_sample = []
    for i in range(2):
        t1 = time.time()
        d = iter(dataloader)
        while True:
            e1.record()
            next(d)
            #for a in dataloader:
            #print(a)
            e2.record()
            e2.synchronize()
            t2 = time.time()
            print("Elapesed time", e1.elapsed_time(e2)/1000)
        average_sample.append(e1.elapsed_time(e2)/1000)
        #average_sample.append(t2-t1)
    average_sample = sum(average_sample[1:])/(len(average_sample)-1)
    return average_sample

def groot_baseline_latency(graph_name, batch_size):
    # graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1,4)
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    gpu_map = [[] for _ in range(4)]
    fanout = 20
    deterministic = False
    testing = False
    self_edge = False
    
    rounds = 1
    pull_optimization = False
    no_layers = 3
    no_gpus = 4
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers, no_gpus)
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
    #graphs = [ "ogbn-products"]

    with open('sampler_perf','a') as fp:
        fp.write("Graph| Batch | Baseline | GRoot\n")
    for graph in agraphs:
        for batch_size in batches:
            t1 = groot_baseline_latency(graph, batch_size)
            t2 = sampler_baseline_latency(graph, batch_size)
            with open('sampler_perf', 'a') as fp:
                fp.write("{}|{}|{}|{}\n".format(graph,batch_size,t2, t1))
    #assert(t1 < t2 * 2)

if __name__=="__main__":
    #test_sampling_overhead()
    graph_name = "ogbn-products"
    batch_size = 4096
    print("Averagge", sampler_baseline_latency(graph_name, batch_size))
    #groot_baseline_latency()
