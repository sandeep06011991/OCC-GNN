import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph

def sampler_baseline_latency():
    graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    train_nids = torch.chunk(train_nid, 4 , dim = 0)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10,10,10])
    dataloaders = [(dgl.dataloading.NodeDataLoader(
        dg_graph,
        train_nids[i],
        sampler,
        device='cpu',
        batch_size= 1024,
        shuffle=True,
        drop_last=True,
        num_workers=0 )) for i in range(4)]
    data_sizes_moved = []
    sampler_and_movenent = []
    for i in range(3):
        t1 = time.time()
        dataloaders_it = [iter(i) for i in dataloaders]
        try:
            while True:
                s =  0
                for j in range(4):
                    a,b,blocks = next(dataloaders_it[j])
                    a = [block.to(torch.device(j)) for block in blocks]
                    for b in blocks:
                        s += b.num_edges() + b.number_of_dst_nodes()
                    data_sizes_moved.append(s)
        except StopIteration:
            pass
        t2 = time.time()
        sampler_and_movenent.append(t2-t1)
    data_movement = sum(data_sizes_moved)/len(data_sizes_moved)
    sampler_and_movement = sum(sampler_and_movement[1:])/2
    return data_movement, sampler_movement


def groot_baseline_latency():
    graph_name = "ogbn-arxiv"
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
    data_sizes_moved = []
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    data_sizes_moved = []
    sampler_and_movenent = []
    for i in range(3):
        j = 0
        t1 = time.time()
        while(j < train_nid.shape[0]):
            csample = slicer.getSample(train_nid[j:j + batch_size].tolist())
            global_sample = Sample(csample)
            local_samples = [Gpu_Local_Sample() for i in range(4)]
            s = 0
            for i in range(4):
                local_samples[i].set_from_global_sample(global_sample, i)
                t  = serialize_to_tensor(local_samples[i])
                s += (t.shape[0])# # print(t.shape)
                t = t.to(torch.device(i))
                construct_from_tensor_on_gpu(t, torch.device(i), local_samples[i])
                # construct_from_tensor_on_gpu(t, torch.device(i), local_samples[i])
                local_samples[i].prepare()
            j = j + batch_size
            data_sizes_moved.append(s)
        t2 = time.time()
        sampler_and_movenent.append(t2-t1)
    return  sum(data_sizes_moved)/len(data_sizes_moved)), sum(sampler_and_movement[1:])/2

def test_data_movement():
    data_moved_1, time_avg_1 = sampler_baseline_latency()
    data_moved_2, time_avg_2 = groot_baseline_latency()
    assert(data_moved_2 < data_moved_1 * 2)
    assert(time_avg_2 < time_avg_1 * 2)

if __name__ == "__main__":
    sampler_baseline_latency()
    groot_baseline_latency()
