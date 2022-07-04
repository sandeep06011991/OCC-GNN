# N: Vertices.
# M: Edges.
# cross: Cross Edges Across all partitions
import dgl
import torch
import subprocess
import numpy as np
from metis import *

TARGET_DIR = "/data/sandeep/synthetic/"

def create_synthetic_dataset(N, M, cross):
    assert(N %4 ==0)
    assert(M %4 ==0)
    assert(cross %4 ==0)
    assert(M < (N*N-1)/2)
    for i in range(4):
        offset = int(i * (N/4))
        if i == 0:
            edges_src,edges_dest  = dgl.rand_graph(int(N/4),int(M/4)).edges()
            continue
        edges =  dgl.rand_graph(int(N/4),int(M/4)).edges()
        edges_src = torch.cat((edges[0] + offset, edges_src),dim = 0)
        edges_dest = torch.cat((edges[1] + offset, edges_dest),dim = 0)
    # print(edges_src, edges_dest)
    graph = dgl.graph((edges_src,edges_dest))
    p_src = torch.randint(0, 4, (cross * 2,))
    p_dest = torch.randint(0, 4, (cross * 2,))
    select = torch.where(p_src != p_dest)[0][:cross]
    print(select.shape , cross)
    assert(select.shape[0] == cross)
    p_src = p_src[select] * int(N/4)
    p_dest = p_dest[select] * int(N/4)
    src = (torch.rand(cross) * N/4).to(int)
    dest = (torch.rand(cross) * N/4).to(int)
    src = src + p_src
    dest = dest + p_dest
    graph.add_edges(src,dest)
    graph = graph.to_simple()
    return graph

    print("Graph creation Done!")
    # Select cross edges

def use_metis():
    pass

def write_dataset(graph, TARGET_DIR):
        # dataset = dgl.add_self_loop(dataset)
    edges = graph.edges()
    dataset = graph
    graph.remove_edges(torch.where(edges[0]==edges[1])[0])
    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape),sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    num_edges = dataset.num_edges()
    num_nodes = dataset.num_nodes()
    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)
    if('train_idx' not in dataset.ndata.keys()):
        nodes = torch.randperm(num_nodes)
        train_idx = nodes[: int(len(nodes)*.8)]
        val_idx = nodes[int(len(nodes)*.8)+1:]
    # train_idx =  dataset[0].ndata['train_idx']
    # val_idx = dataset[0].ndata['val_idx']
    meta = {}
    # with open(TARGET_DIR + '/partition_map.bin','wb') as fp:
    #     fp.write(p_map.numpy().astype(np.int32).tobytes())
    with open(TARGET_DIR+'/indptr.bin','wb') as fp:
        fp.write(indptr.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indices.bin','wb') as fp:
        fp.write(indices.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/train_idx.bin','wb') as fp:
        fp.write(train_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/val_idx.bin','wb') as fp:
        fp.write(val_idx.numpy().astype('int64').tobytes())
    csum_train = torch.sum(train_idx).item()
    csum_test = torch.sum(val_idx).item()

    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    # meta_structure["csum_offsets"] = csum_offsets
    # meta_structure["csum_edges"] = csum_edges
    # meta_structure["num_classes"] = num_classes
    with open(TARGET_DIR+'/meta.txt','w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k,meta_structure[k]))
    print("All data written!")

    # final graph is here graph
    # Check metis partition.
    # Run experiment show decrease in communication over pagrahself.

N,M,C = 200,1000, 400
g = create_synthetic_dataset(N,M,C)
import os
graphname = "synth_{}_{}_{}".format(N,M,C)
TARGET_DIR = TARGET_DIR + graphname
os.makedirs(TARGET_DIR,exist_ok = True)
write_dataset(g, TARGET_DIR)
metis_graphname = "synthetic/{}".format(graphname)
create_metis_file(metis_graphname)
run_metis(metis_graphname)
#
# if __name__ == "__main__":
#     create_synthetic_dataset(20,100,20)
