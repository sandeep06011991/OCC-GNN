import dgl
import torch
from dgl._deprecate.graph import DGLGraph
from dgl.contrib.graph_store import create_graph_store_server
from dgl.contrib.graph_store import create_graph_from_store
import time
# Simple test to see how graph server works

graph_data =  DGLGraph(([1,2,3,4],[2,34,5,6]))
graph_data.readonly()
graph_name = "graph-dummy"
store_type = "shared_mem"
num_workers = 3
g1 = create_graph_store_server(graph_data, graph_name, store_type, num_workers,
                              multigraph=False, port=8000)

g1.ndata['in'] = torch.rand(35,3)

# g1.set_n_repr({'in':torch.rand(35,3)})
# g1.run()

def func(g1):
    g1.run()
    print("Server created")

import threading
t1 = threading.Thread(target = func,args = (g1,))
t1.start()

time.sleep(2)
g2 = create_graph_from_store(graph_name , store_type, port=8000)
nid = torch.tensor([0,1,2])
print(g2._node_frame._frame["in"].data[nid].shape)
print("client accesses it")
