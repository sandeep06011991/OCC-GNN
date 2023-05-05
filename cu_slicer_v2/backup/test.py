from utils.utils import get_process_graph
import cuslicer
import torch
import dgl
import time

def naive_three_layer_sample(graph_name):
    dgl_graph, p_map, classes = get_process_graph(graph_name, -1)
    indptr, indices, edges= dgl_graph.adj_sparse('csc')
    print(indptr.shape, indices.shape, indptr.dtype)
    indptr = indptr.to(1)
    indices = indices.to(1)
    torch.cuda.set_device(1)
    t_nodes = torch.arange(p_map.shape[0],device = 1)
    for k in range(3):
        b = 0
        t1 = time.time()
        while(b < p_map.shape[0]):
            seeds = t_nodes[b:b+4096].clone()
            for i in range(3):
                bipartite_graph = cuslicer.sample_layer(indptr,indices,seeds);
                torch.cuda.synchronize()
        #print(bipartite_graph.a, bipartite_graph.b)
                seeds =bipartite_graph.c
            b = b + 4096
        #print(bipartite_graph.b.shape, bipartite_graph.b.dtype, i)
        #vals = torch.sort(bipartite_graph.b)[0]
        #seeds = vals
        print("Custom time", time.time() - t1)
        #seeds = torch.unique(torch.randint(0,1000,(10,),device = 1))
        #print(seeds.shape, seeds.dtype)
    print("All done")

def dgl_sampler(graph_name):
    dgl_graph, p_map, classes = get_process_graph(graph_name, -1)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3);

    world_size = 4
    train_nid = torch.arange(p_map.shape[0], device = 1)

    number_of_minibatches = train_nid.shape[0]/4096
    
    g = dgl_graph.to(1)

    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cuda',
        batch_size= 4096,
        shuffle=False,
        drop_last = False,
        num_workers = 0)

 
    t1 = time.time()
    for i in range(3):
        t1 = time.time()
        for i in dataloader:
            pass
        t2 = time.time()    
        print("DGL Time", t2-t1)    


if __name__ == "__main__":
    graph_name = "ogbn-products"
    #naive_three_layer_sample(graph_name)
    dgl_sampler(graph_name)


