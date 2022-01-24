import torch
import time
import dgl
import numpy as np
import scipy
from dgl._deprecate.graph import DGLGraph as DGLGraph
from dgl.contrib.sampling import NeighborSampler as NeighborSampler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from dgl.nn import GraphConv
import dgl.function as fn
import time

def get_graph():
    DATA_DIR = "/home/spolisetty/data"
    graphname = "ogbn-products"
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.intc)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.intc)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    fsize = 1024
    features = torch.rand(num_nodes,fsize)
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
    shape = (num_nodes,num_nodes))
    dg_graph = DGLGraph(sp)
    dg_graph.ndata["features"] = features
    dg_graph.readonly()
    p_map = np.fromfile("{}/{}/partition_map.bin".format(DATA_DIR,graphname),dtype = np.intc)
    edges = dg_graph.edges()
    partition_map = torch.from_numpy(p_map)
    return dg_graph, partition_map, features, num_nodes, num_edges



class Model(torch.nn.Module):
    def __init__(self,fsize,device_id):
        super(Model, self).__init__()
        # self.conv1 = GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True)
        # self.conv2 = GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True)
        self.device_id = device_id
        self.movement_time = []
        self.compute_time = []

    def forward(self,nf):
        t1 = time.time()
        nf.copy_from_parent(ctx = torch.device(self.device_id))
        t2 = time.time()
        self.movement_time.append(t2-t1)
        nf.block_compute(0 \
        , fn.copy_src(src='features', out='m'),
                                 lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                                 )
        nf.block_compute(1 \
        , fn.copy_src(src='h', out='m'),
                                 lambda node : {'h2': node.mailbox['m'].mean(dim=1)},
                                 )
        t3 = time.time()
        self.compute_time.append(t3-t2)
        return nf

def train(rank, world_size):
    hops = 2
    fsize = 1024
    gpu_id = rank
    dg_graph,partition_map, features,num_nodes, num_edges = get_graph()
    # return
    # net = torch.nn.DataParallel(Model(fsize), device_ids=[0, 1, 2,3]).cuda()
    per_batch = int(num_nodes/4)
    models = Model(fsize,gpu_id).to(torch.device(gpu_id))
    # dg_graph = dg_graph.to('cpu')
    samplers = NeighborSampler(
            dg_graph, 1024, expand_factor = 10, num_hops = hops, \
            seed_nodes = torch.arange(per_batch * gpu_id, per_batch * gpu_id + 1,1) , \
            shuffle = True)
    # t1 = time.time()
    for nf in samplers:
        x = models(nf)
    # t2 = time.time()
    if rank ==0:
        print("data movement time:", sum(models.movement_time))
        print("compute time:",sum(models.compute_time))

if __name__=="__main__":
    gpu_num = [0,1,2,3]
    mp.spawn(train, args=(gpu_num,), nprocs=len(gpu_num), join=True)
    print("All done!")
