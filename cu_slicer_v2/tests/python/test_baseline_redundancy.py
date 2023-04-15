from utils.utils import get_process_graph
import dgl
import torch
import time
import nvtx

def partition_load(nodes, partition, k):
    load = []
    total_nodes = nodes.shape[0]
    for i in range(k):
        load.append(torch.sum(partition[nodes] == i)/total_nodes)
    print(load)

def edges_per_epoch(filename, batch_size):
    device = 'cpu'
    #torch.cuda.set_device(device)
    # filename = "ogbn-arxiv"
    
    num_gpus = 4

    dg_graph, partition_map, num_classes =\
        get_process_graph(filename, -1 ,  num_gpus)
    fsize = dg_graph.ndata['features'].shape[1]
    g = dg_graph.to(device)
    p_map = partition_map
    #partition = dgl.metis_partition(dg_graph,4, balance_edges = False, reshuffle = False)
    #p_map = torch.zeros((g.num_nodes(),))
    #for k in partition.keys():
    #    p_map[partition[k].ndata['_ID']] = k

    print(" Device")
    #print("number of nodes", g.num_nodes())
    #train_nid = torch.arange(0,g.num_nodes(), device = device)
    train_nid = g.ndata['train_mask'].nonzero().flatten()
    print(len(train_nid)/batch_size,"total number of batches")

    num_neighbors = 20
    # sampler = dgl.dataloading.NeighborSampler(
    #     [15,10,5], replace = True)

    sampler = dgl.dataloading.NeighborSampler(
        [int(num_neighbors) for i in range(3)], replace = False)
    dataloader = dgl.dataloading.NodeDataLoader(\
        g,\
        train_nid,\
        sampler,\
        device=device,\
        batch_size=batch_size,\
        shuffle=True,\
        drop_last=True)
    dt = iter(dataloader)
    i = 0
    t1 = time.time()
    total_edges = 0
    try:
        while True:
            t1 = time.time()
            with nvtx.annotate("Sample",color = 'blue'):
                input_, output, blocks = next(dt)
                edges = []
                nodes = []
                nodes.append(input_.shape[0])
                partition_load(input_, p_map, 4)
                for b in blocks:
                    edges.append(b.num_edges())
                    nodes.append(b.num_dst_nodes())
                    total_edges += (b.num_edges())
                print("Nodes", nodes , "Edges", edges)

            t2 = time.time()
            # print("Total sampling time", t2-t1)
    except StopIteration:
        pass
    return total_edges


# Ogbn-products with a fanout of 20 and three layers
# bs 1024 edges 8319487800
# bs 4096 edges 6621680480 0.7959240567670525
# bs 16384 edges 3979739620 0.601016559470112
def test_redundancy_in_edges():
    graph_name = "ogbn-products"
    e1 = edges_per_epoch(graph_name, 1024)

    e2 = edges_per_epoch(graph_name, 4096)

    #e3 = edges_per_epoch(graph_name, 4096)

    print("bs",1024, "edges", e1)
    print("bs",4*1024, "edges", e2, e2/e1)
    #print("bs",4*4 *1024, "edges", e3, e3/e2)

# b = test_sampling_overhead()
if __name__=="__main__":
    t = test_redundancy_in_edges()
