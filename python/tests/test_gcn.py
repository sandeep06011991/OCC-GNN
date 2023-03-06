import os
import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph
from layers.dist_sageconv import *
import torch.distributed as dist
from cuslicer import cuslicer
from data import Bipartite, Sample, Gpu_Local_Sample
from models.dist_gcn import get_sage_distributed
from models.dist_gat import get_gat_distributed


def trainer(proc_id, world_num):
    graph_name = "ogbn-arxiv"
    torch.cuda.set_device(proc_id)
    backend = "nccl"
    rank = proc_id
    world_size = world_num
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    num_hidden = 16
    deterministic = True
    model = "gcn"
    gpus = 4
    num_layers = 3
    dg_graph, p_map, num_classes = get_process_graph(graph_name, -1, gpus)
    features = dg_graph.ndata['features']        
    Model = get_sage_distributed(num_hidden, features, num_classes,
            proc_id, deterministic, model, gpus,  num_layers)
    self_edge = False
    attention = False
    pull_optimization = False
    model = Model.to(proc_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


    ## CUSLICER GET CU SAMPLE 
    
    num_layers = 3
    num_gpus = 4

    storage_map_part = [[],[],[],[]]
    csl3 = cuslicer(graph_name, storage_map_part,
        [-1, -1, -1], False , False, True, 4, False, num_layers, num_gpus, proc_id)
    
    csample =  csl3.getSample([i for i in range(4096)])
    tensorized_sample = Sample(csample)
    obj = Gpu_Local_Sample()
    attention = False
    obj.set_from_global_sample(tensorized_sample, proc_id)
    obj.prepare(attention)
    
    input_data = torch.ones((obj.cache_miss_to.shape[0], features.shape[1]), device = proc_id)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    for i in range(10):
        e1.record()
        print("Start")
        torch.cuda.nvtx.range_push("culice_minibatch")
        out = model(obj, input_data, None)
        torch.cuda.nvtx.range_pop()
        e2.record()
        e2.synchronize()
        print("time", e1.elapsed_time(e2)/1000)
    print("Forward pass done!")
'''
time 0.009733119964599609
time 0.009049087524414063
time 0.009588735580444336
time 0.009523200035095216
time 0.009492480278015136
time 0.009554944038391112
'''
def test_groot_gcn():
    gpu_num = 4
    
    mp.set_start_method('spawn')
    p = []
    for i in range(gpu_num):
        pp = mp.Process(target = trainer, args=(i, gpu_num))
        pp.start()
        p.append(pp)
    for pp in p:
        pp.join()
'''
forward pass time 0.003757055997848511
forward pass time 0.003738624095916748
forward pass time 0.003949568033218383
forward pass time 0.0037068800926208494
forward pass time 0.0037376000881195067
forward pass time 0.0037201919555664062
forward pass time 0.004220928192138672'''
def get_correct_gcn():
    graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1, 4)
    train_nid = torch.arange(4096 * 4)
    #dg_graph = dg_graph.add_self_loop()
    num_layers = 3
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    #sampler = dgl.dataloading.NeighborSampler([20,20,20])
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph.to(0),
        train_nid.to(0),
        sampler,
        device='cuda',
        batch_size= 4096,
        shuffle=False,
        drop_last= False,
        num_workers=0 )
    layers = []
    layers.append(dgl.nn.SAGEConv(128, 16, aggregator_type = 'mean').to(0))
    for i in range(num_layers - 2):
        layers.append(dgl.nn.SAGEConv(16,16, aggregator_type = 'mean').to(0))
    layers.append(dgl.nn.SAGEConv(16,40, aggregator_type = 'mean').to(0))    
    
    
    it = iter(dataloader)
    input_nodes, seeds, blocks = next(it)
    print(seeds.shape,"batch_size")
    for l in layers:
        l.fc_self.weight = torch.nn.Parameter(torch.ones(l.fc_self.weight.shape, device = 0))
        l.fc_neigh.weight = torch.nn.Parameter(torch.ones(l.fc_neigh.weight.shape, device = 0))
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)

    for input_nodes, seeds, blocks in dataloader:
        c  =  torch.ones(input_nodes.shape[0],128, device = 0)
        
        for k in range(10):
            b = c
            e1.record()
            torch.cuda.nvtx.range_push("minibatch-naive")
            for i,l in enumerate(layers):
                b = l(blocks[i], b)
            #print(torch.sum(b))
            torch.cuda.nvtx.range_pop()
            e2.record()
            e2.synchronize()
            print("forward pass time", e1.elapsed_time(e2)/1000)
        torch.sum(b).backward()
    
# get_correct_gat()
    # test_heterograph_construction_python()

if __name__ == "__main__":
    print(get_correct_gcn())
    #print("Seperate")
    print(test_groot_gcn())
