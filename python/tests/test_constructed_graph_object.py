import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph

def test_heterograph_construction_python():
    graph_name = "ogbn-arxiv"
    gpu_map = [[] for _ in range(4)]
    fanout = 1000
    deterministic = True
    testing = False
    self_edge = False
    rounds = 1
    pull_optimization = False
    no_layers = 1
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    csample = slicer.getSample([i for i in range(10000)])
    global_sample = Sample(csample)
    local_samples = [Gpu_Local_Sample() for i in range(4)]
    for i in range(4):
        local_samples[i].set_from_global_sample(global_sample, i)
        # local_samples[i].prepare()
    # serialize_to_tensor(global_sample)
    print(local_samples[i].layers[0].indptr_R.shape,"Cross check !!!")
    print(local_samples[i].layers[0].num_out_remote,"Cross check !!!")

    # x = torch.sum(local_samples[i].layers[0].indptr_L)
    t  = serialize_to_tensor(local_samples[i])
    object = Gpu_Local_Sample()
    t = t.to(3)
    construct_from_tensor_on_gpu(t, torch.device(3),  object)
    print(object.layers[0].indptr_R.shape,"Cross check !!!")
    print(object.layers[0].num_out_remote,"Cross check !!!")
    print(object)
    # assert(x == y)
    object.prepare()
    bp_graph = object.layers[0]
    print(bp_graph)
    a = bp_graph.num_in_nodes_local
    features = torch.rand(a,100, device = torch.device(3))

    torch.cuda.set_device(3)
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    # Initialise cuda tensors here. E.g.:
    for i in range(10):
        t1 = time.time()
        e1.record()
        with torch.cuda.stream(s1):
            d = bp_graph.gather_remote(features)
            e = d.to('cpu')
        with torch.cuda.stream(s2):
            c = bp_graph.gather_local(features)
    # # Wait for C and D to be computed.
        s1.synchronize()
        # s2.synchronize()
        e2.record()
        e2.synchronize()

        print("Overlaping", e1.elapsed_time(e2)/1000)

        # print(torch.sum(e))
        # print(torch.sum(c))
        # t2 = time.time()
        # print("Overlapped time",t2-t1)
        # print(features.shape)
        e0.record()
        # e1.record()
        d = bp_graph.gather_remote(features)
        # e2.record()
        # e2.synchronize()
        # print("k1", e1.elapsed_time(e2)/1000)
        # e1.record()
        f = d.to('cpu')
        # e2.record()
        # e2.synchronize()
        # print("M1", e1.elapsed_time(e2)/1000)
        # e1.record()
        c = bp_graph.gather_remote(features)
        e2.record()
        e2.synchronize()
        print("Sqeuntial ", e1.elapsed_time(e2)/1000)
        # print(torch.sum(f))
        # print(torch.sum(c))
        # e2.record()
        # e2.synchronize()
        # print("sequential time", e1.elapsed_time(e2)/1000)
        # Overlapping works What next





def get_correct_gnn():
    graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = torch.arange(1000)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        train_nid,
        sampler,
        device='cpu',
        batch_size= 1000,
        shuffle=True,
        drop_last=True,
        num_workers=0 )
    nn = dgl.nn.SAGEConv(10, 10,  'mean')
    it = iter(dataloader)
    input_nodes, seeds, blocks = next(it)
    nn.weight = torch.nn.Parameter(torch.ones(10,10))
    f_out = nn(blocks[0], torch.ones(input_nodes.shape[0],10))
    print(f_out)

def sampler_baseline_latency():
    graph_name = "ogbn-arxiv"
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
    for i in range(3):
        t1 = time.time()
        for _ in dataloader:
            pass
        t2 = time.time()
        print("Sampler total time",t2-t1)

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
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    for i in range(3):
        j = 0
        t1 = time.time()
        while(j < train_nid.shape[0]):
            csample = slicer.getSample(train_nid[j:j + batch_size].tolist())
            j = j + batch_size
        t2 = time.time()
        print("GRoot Sampler",t2 - t1)
if __name__=="__main__":
    sampler_baseline_latency()
    groot_baseline_latency()
    # get_correct_gnn()
    # test_heterograph_construction_python()
