
# Read next door graphs and create CSR.
from ctypes import *
from ctypes.util import *
import networkx as nx, numpy as np

l1 = '/mnt/homes/spolisetty/nextdoor-experiments/graph_loading/libgraph.so'
#l1 = 'libGraph.so'
# print(find_library(l1))
d = CDLL(l1)
d.loadgraph.argtypes = [c_char_p]


# graphPath = create_string_buffer("~/GPUesque/input/ppi.data")
graphPath = "/mnt/homes/spolisetty/NextDoor/input/ppi.data"
graphPath = bytes(graphPath, encoding='utf8')
d.loadgraph(graphPath)
print("Graph Loaded in C++")

d.getEdgePairList.restype = np.ctypeslib.ndpointer(dtype=c_int, shape=(d.numberOfEdges(), 2))
edges = d.getEdgePairList()

G = nx.Graph()
print("Loading networkx graph")
G.add_edges_from(edges)

def get_graph_from_nextdoor_datapath(filename):
    pass
    
