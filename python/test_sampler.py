import dgl
import torch
import time
from dgl.sampling import sample_neighbors
from utils.utils import get_process_graph
graph,features,_ = get_process_graph('ogbn-products')
batch_size = 4096

graph = graph
batch = torch.arange(graph.num_nodes())
def sample(graph, batch_in):
    layers = []
    blocks = []
    # Note. Dont forget self loops otherwise GAT doesnt work.
    # Create bipartite graphs for the first l-1 layers
    last_layer = batch_in
    last_layer = last_layer.unique()
    layers.append(last_layer)
    fanouts = [10,10]
    for fanout in fanouts:
        dest_ids,src_ids = sample_neighbors(graph, last_layer, fanout).edges()
        self_loop_dests = torch.cat([last_layer, dest_ids])
        edges = self_loop_dests, torch.cat([last_layer, src_ids])
        last_layer = torch.unique(self_loop_dests)
        layers.append(last_layer)
        blocks.append(edges)
    return layers,blocks

t1 = time.time()
for offset in range(0,batch.shape[0],batch_size):
    train_idx = batch[offset:offset + batch_size]
    _ = sample(graph, train_idx )
t2 = time.time()

print("gpu sampling time", t2 - t1)

print("all sampling success")
