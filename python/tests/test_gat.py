import os
import torch
from cslicer import cslicer
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
from utils.utils import get_process_graph
from layers.dist_gatconv import *
import torch.distributed as dist

class Model(nn.Module):

    def __init__(self, gpu_id):
        super().__init__()
        self.gat_conv1 = DistGATConv(10, 10, gpu_id, num_heads = 1, deterministic = True)
        self.gat_conv2 = DistGATConv(10, 10, gpu_id, num_heads = 1, deterministic = True)

    def forward(self, local_graph, x):
        l1 = self.gat_conv1(local_graph.layers[0], x, 0)
        print(l1.shape)
        l2 = self.gat_conv2(local_graph.layers[1], l1, 1)
        return l2

def trainer(proc_id, world_num):
    backend = "nccl"
    rank = proc_id
    world_size = world_num
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = Model(proc_id).to(proc_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    graph_name = "ogbn-arxiv"
    dg_graph, partition_map, _ = get_process_graph(graph_name, -1)
    gpu_map = [[] for _ in range(4)]
    fanout = 1000
    deterministic = True
    testing = False
    self_edge = True
    rounds = 4
    pull_optimization = False
    no_layers = 2
    slicer = cslicer(graph_name, gpu_map, fanout,
       deterministic, testing,
          self_edge, rounds, pull_optimization, no_layers)
    train_nids = torch.arange(1)
    csample = slicer.getSample(train_nids.tolist())
    print(train_nids, "Train")
    global_sample = Sample(csample)

    local_samples = [Gpu_Local_Sample() for i in range(4)]
    local_samples[proc_id].set_from_global_sample(global_sample, proc_id)

    # x = torch.sum(local_samples[i].layers[0].indptr_L)
    t  = serialize_to_tensor(local_samples[proc_id])
    bp_object = Gpu_Local_Sample()
    t = t.to(proc_id)
    construct_from_tensor_on_gpu(t, torch.device(proc_id),  object)
    bp_object.prepare(attention = True)
    n = bp_object.cache_hit_from.shape[0] + bp_object.cache_miss_from.shape[0]
    n1 = torch.ones(n, 10, device = proc_id, requires_grad = True)
    #print(n1.shape, object.cache_miss_from)
    n = n1 * (bp_object.cache_miss_from.reshape(bp_object.cache_miss_from.shape[0],1) % 10)
    print(bp_object.layers[0].push_to_ids)
    #out = model(object, n )
    #print(torch.sum(out), "my out")
    #torch.sum(out).backward()
    #print(torch.sum(n1.grad),"my grad")

def test_groot_gat():
    gpu_num = 4
    # mp.set_start_method('spawn')
    mp.spawn(trainer, args=(gpu_num,), nprocs=gpu_num, join=True)


def get_correct_gat():
    graph_name = "ogbn-arxiv"
    dg_graph, partition_map, num_classes  = get_process_graph(graph_name, -1)
    train_nid = torch.arange(1)
    dg_graph = dg_graph.add_self_loop()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        train_nid,
        sampler,
        device='cpu',
        batch_size= 1000,
        shuffle=True,
        drop_last= False,
        num_workers=0 )
    nn1 = dgl.nn.GATConv(10, 10, 1)
    nn2 = dgl.nn.GATConv(10 ,10 ,1)
    it = iter(dataloader)
    input_nodes, seeds, blocks = next(it)
    print(seeds.shape,"batch_size")
    nn1.fc.weight = torch.nn.Parameter(torch.ones(nn1.fc.weight.shape))
    nn1.attn_r = torch.nn.Parameter(torch.ones(nn1.attn_r.shape))
    nn1.attn_l = torch.nn.Parameter(torch.ones(nn1.attn_l.shape))
    nn2.fc.weight = torch.nn.Parameter(torch.ones(nn2.fc.weight.shape))
    nn2.attn_r = torch.nn.Parameter(torch.ones(nn2.attn_r.shape))
    nn2.attn_l = torch.nn.Parameter(torch.ones(nn2.attn_l.shape))

    a  =  torch.ones(input_nodes.shape[0],10, requires_grad = True) 
    f_out = nn1(blocks[0], (input_nodes.reshape(input_nodes.shape[0],1) % 10) * a)
    print(f_out.shape)
    x = nn2(blocks[1], f_out)
    print(torch.sum(x))
    torch.sum(x).backward()
    for i in range(4):
        print(torch.sum(x[torch.where(partition_map[seeds] ==i)[0]]), i, " correct out")
        print(torch.sum(a.grad[torch.where(partition_map[input_nodes]==i)[0]]), i , "correct grad")

# get_correct_gat()
    # test_heterograph_construction_python()

if __name__ == "__main__":
    #get_correct_gat()
    test_groot_gat()
