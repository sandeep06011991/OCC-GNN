import dgl
import torch
from dgl.sampling import sample_neighbors

def get_subgraph_partition(g1, W , gpu):
    # g1 = dgl.rand_graph(1000,10000)
    # g1.to_simple()
    v,u = g1.edges()
    select =  torch.where(W[u] == gpu)
    subgraph = dgl.graph((v[select],u[select]))
    return subgraph

# Finish pseudo code of pure W layer based sampling.
def sample_local_graph(gs, training_nodes, W, sf):
    edges = []
    for i in range(4):
        v,u = sample_neighbors(gs[i], training_nodes[i], 10).edges()
        edges.append((v,u))
    for i in range(4):
        local_edges = (W[v] == i)
        for j in range(4):
            remote_edges = (edges[j][0][W[v]]  == i)
            create_local_graph
            create_shuffle_indices

def local_sampler():
    pass

g1 = dgl.rand_graph(1000,10000)
g1.to_simple()
W = torch.randint(0,4,(1000,))
g = get_subgraph_partition(g1,W,0)


assert(False)
