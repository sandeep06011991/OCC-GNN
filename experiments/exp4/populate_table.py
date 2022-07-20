# Run a simulation
# Measure total communication cost for various kinds of samplers assuming a data local format
# stick to node neighbour sampling
# Graph | sampling |
#  comm-pagraph | comm-me |
#   redundant-comp pagraph | redundant-me |
#     skew-pagraph | skew me
# Todos:
# 1. create dgl graph from numpy array (DONE)
# 2. Load partition map
# 3. Creating training vertices arrays for 4 partitions

DATA_DIR = "/home/spolisetty/data"

import sys
import dgl
import torch
import numpy as np
import dgl
import scipy
from dgl._deprecate.graph import DGLGraph as DGLGraph
from dgl.contrib.sampling import NeighborSampler as NeighborSampler

# nf.map_to_parent_eid(0)
# nf.map_to_parent_nid(0)
# nf.block_edges
# Todo take average instead of abo
def run_experiment(graphname,cache_percentage,hops):

    # hops = 4
    # cache_percentage = .10
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.intc)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.intc)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),shape = (num_nodes,num_nodes))
    dg_graph = DGLGraph(sp)
    dg_graph.readonly()
    p_map = np.fromfile("{}/{}/partition_map.bin".format(DATA_DIR,graphname),dtype = np.intc)
    edges = dg_graph.edges()
    partition_map = torch.from_numpy(p_map)
    internal_edge_bool = partition_map[edges[0]] == partition_map[edges[1]]
    print("internal edges",torch.sum(internal_edge_bool),"out of ",num_edges)
    in_d = dg_graph.in_degrees()
    values, indices = in_d.topk(int(num_nodes  * cache_percentage))
    pa_cache = torch.zeros(num_nodes)
    pa_cache[indices] = True
    if(cache_percentage * 4 > .99):
        my_cache_percentage = 1
        my_cache = torch.ones(num_nodes)
    else:
        my_cache_percentage = cache_percentage * 4
        values, indices = in_d.topk(int(num_nodes  * my_cache_percentage))
        my_cache = torch.zeros(num_nodes)
        my_cache[indices] = True
    cross_edge_bool = ~ internal_edge_bool
    assert(p_map.shape[0] == num_nodes)
    training_nodes = []
    for i in range(4):
        # print(partition_map == i)
        training_nodes.append((partition_map==i).nonzero(as_tuple=True)[0])

    dataloaders = []
    for i in range(4):
        # sampler = dgl.dataloading.MultiLayerNeighborSampler([10 for i in range(hops)])
        dataloader = iter(NeighborSampler(
            dg_graph, 1024, expand_factor = 10, num_hops = hops,
            seed_nodes = training_nodes[i], shuffle = True))
        dataloaders.append(dataloader)
    # dataloader_full = iter(NeighborSampler(
    #     dg_graph,4096, expand_factor = 10, num_hops = hops,
    #     seed_nodes = dg_graph.nodes(), shuffle = True))

    epoch = 0
    while True:
        epoch = epoch + 1
        # Iterate through all samplers
        in_all = [0,0,0,0]
        out_all = [0,0,0,0]
        in_size = [0,0,0,0]
        nodeflow_all = [None,None,None,None]
        for i in range(4):
            try:
                o = next(dataloaders[i])
                in_all[i] = o.layer_parent_nid(0)
                out_all[i] = o.layer_parent_nid(-1)
                nodeflow_all[i] = o
                in_size[i] = in_all[i].shape[0]
            except StopIteration:
                pass
        # print(in_size)
        if   any([i ==0  for i in in_size]):
            break
        # Pagraph nodes that have to be nodes_moved
        pagraph = 0
        pagraph_t = 0
        pa_cache_saving = 0
        my_cache_saving = 0
        # if epoch <10:
        #     # warm up
        #     continue

        for i in range(4):
            # print(pa_cache_saving)
            pa_cache_saving = pa_cache_saving + pa_cache[in_all[i]].nonzero().shape[0]
            my_cache_saving = my_cache_saving + my_cache[in_all[i]].nonzero().shape[0]
            pagraph = pagraph + torch.sum(partition_map[in_all[i]]!=i)
            pagraph_t = pagraph_t + in_all[i].shape[0]
        # Me our communication
        me_comm = 0
        me_comm1 = 0
        for i in range(4):
            nf = nodeflow_all[i]
            edges = torch.arange(nf.number_of_edges())
            real_edges = nf.map_to_parent_eid(edges)
            cross_edges_all_block = cross_edge_bool[real_edges]
            cross_edge_block = cross_edges_all_block.nonzero(as_tuple=True)[0]
            for b in range(nf.num_blocks):
                src, dst, edge_id = nf.block_edges(b)
                common_edge = np.intersect1d(edge_id, cross_edge_block)
                # print(edge_id)
                edge = common_edge - edge_id[0].item()
                me_comm1 = me_comm1 + torch.unique(dst[edge]).shape[0]
            # e_1 = nf.edges()
            # print(e_1)
            # d = e_1cross_edge_bool[real_edges].nonzero()
            me_comm  = me_comm + torch.sum(cross_edge_bool[real_edges])
        # redundant in pagraph
        pa_red = 0
        pa_t = 0
        for i in range(4):
            b_i = nodeflow_all[i]
            # print(b_i.num_layers)
            for h in range(hops+1):
                if h==0:
                    continue
                pa_t = pa_t + b_i.layer_nid(h).shape[0]
                for j in range(i+1,4):
                    if i==j:
                        continue
                    b_j = nodeflow_all[j]
                    a = b_i.layer_parent_nid(h)
                    b = b_j.layer_parent_nid(h)
                    t1 = np.intersect1d(a,b)
                    t1  = t1.shape[0]
                    pa_red = pa_red + t1
        # print(pa_red,pa_t)
        # Skew in pagraph
        work = [0,0,0,0]
        for i in range(4):
            work[i] = nodeflow_all[i].number_of_edges()
        skew = (max(work) - min(work))/min(work)
        # print("{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.4f}|{:.2f}".format(
        # pagraph/pagraph_t , me_comm/pagraph_t , \
        #     me_comm1/pagraph_t, \
        #         pa_cache_saving/pagraph_t, my_cache_saving/pagraph_t,\
        #             pa_red/( pa_t), skew))
        return pagraph/pagraph_t , me_comm/pagraph_t , \
            me_comm1/pagraph_t, \
                pa_cache_saving/pagraph_t, my_cache_saving/pagraph_t,\
                    pa_red/(pa_t), skew


    print("hello world")
    # partition_id
    # Create a 4 layer
def populate_table():
    graph_names = ["ogbn-arxiv","ogbn-products","reddit"]
    graph_names = ["ogbn-products"]
    cache_percentage = [.05]
    hops = [2,3,4]
    hops = [4]
    with open("exp4.txt",'a') as fp:
        fp.write("graph|hops|naive-partition|cross-edge-comm|cross-node|pa-cache|my-cache|red|skew\n")
    for h in hops:
        for gn in graph_names:
            for c in cache_percentage:
                (a,b,c,d,e,f,g) = run_experiment(gn,c,h)
            with open("exp4.txt",'a') as fp:
                fp.write("{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f} \n".\
                    format(gn,h, a,b,c,d,e,f,g))



if __name__=="__main__":
    populate_table()
