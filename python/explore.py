import torch
import time
import dgl
import numpy as np
import scipy
from dgl import DGLGraph as DGLGraph
from dgl.contrib.sampling import NeighborSampler as NeighborSampler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from dgl.nn import GraphConv
import dgl.function as fn
import time



DATA_DIR = "/home/spolisetty/data"
graphname = "ogbn-arxiv"
indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.intc)
indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.intc)
num_nodes = indptr.shape[0] - 1
num_edges = indices.shape[0]
fsize = 1024
features = torch.rand(num_nodes,fsize)
sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
shape = (num_nodes,num_nodes))
dg_graph = DGLGraph(sp)
# dg_graph.ndata["features"] = features
dg_graph.readonly()
p_map = np.fromfile("{}/{}/partition_map.bin".format(DATA_DIR,graphname),dtype = np.intc)
edges = dg_graph.edges()
partition_map = torch.from_numpy(p_map)
fan_out = "10,10,10"
train_nid = torch.arange(num_nodes)
sampler = dgl.dataloading.MultiLayerNeighborSampler(
    [int(fanout) for fanout in fan_out.split(',')])

dataloader = dgl.dataloading.NodeDataLoader(
    dg_graph,
    train_nid,
    sampler,
    batch_size=4096,
    shuffle=True,
    drop_last=False,
    num_workers=1)

# a = torch.rand(1000*1000*1000).to('cuda')
# del a
#
# import time

# for i in range(100):
#     t_0 = time.time()
in_nodes, out_nodes, blocks = next(iter(dataloader))
    # t1 = time.time()
    # # print(blocks[0].ndata)
    # # assert(blocks[0].ndata["features"]["_U"].device == torch.device("cpu"))
    # # blocks[0].ndata["features"]["_U"].to("cuda")
    # t2 = time.time()
    # # features[blocks[0].ndata["_ID"]["_U"]].to("cuda")
    # t3 = time.time()
    # print("Sampling")
    # print(t1-t_0)
    # print(t2-t1)
    # print(t3-t1)
# in_nodes in graph format
# out_nodes in graph format
# blocks[0]
# Blocks contain
# Block(num_src_nodes=49276, num_dst_nodes=31027, num_edges=110110)
# return dg_graph, partition_map, features, num_nodes, num_edges
# Questions:
# Where is Block Features
