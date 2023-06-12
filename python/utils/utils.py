#### Miscellaneous functions

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
#
# TODO: confirm if this is necessary for MXNet and Tensorflow.  If so, we need
# to standardize worker process creation since our operators are implemented with
# OpenMP.

import scipy.sparse
import torch.multiprocessing as mp
from _thread import start_new_thread
from functools import wraps
import traceback
import numpy as np
import torch
from dgl import DGLGraph
import dgl
from os.path import exists
import logging 
import time 
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import os
from os.path import exists
import dgl
import time 

def get_data_dir():
    # Todo: Repeated code, 
    # Remove this and use the same function in data/env.py
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
        PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN/python"
        ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
        SYSTEM = "jupiter"
        PATH_DIR = "/home/spolisetty/OCC-GNN/python"
        ROOT_DIR = "/home/spolisetty/OCC-GNN"
        OUT_DIR = '/home/spolisetty/OCC-GNN/experiments'
        PA_ROOT_DIR = "/home/spolisetty/OCC-GNN/upgraded_pagraph"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
        PATH_DIR = "/home/q91/OCC-GNN/python"
    if username == "ubuntu":
        DATA_DIR = "/home/ubuntu/data"
        # DATA_DIR = "/data"
        SYSTEM = "P4"
        PATH_DIR = "/home/ubuntu/OCC-GNN/python"
        ROOT_DIR = "/home/ubuntu/OCC-GNN"
        OUT_DIR = '/home/ubuntu/OCC-GNN/experiments'
        PA_ROOT_DIR = "/home/ubuntu/OCC-GNN/upgraded_pagraph"

    return DATA_DIR,PATH_DIR, ROOT_DIR, OUT_DIR, SYSTEM, PA_ROOT_DIR

DATA_DIR,PATH_DIR,ROOT_DIR, OUT_DIR ,SYSTEM, PA_ROOT_DIR = get_data_dir()

def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

synthetic_graphs = ["com-youtube","com-orkut","com-friendster"]

def read_meta_file(filename):
    with open("{}/{}/meta.txt".format(DATA_DIR, filename)) as fp:
        lines = fp.readlines()
        results = {}
        for l in lines:
            k,v = l.split("=")
            if k in ["num_nodes","num_edges", "num_classes" , "feature_dim"]:
                v = int(v)
            results[k] = v
    return results

def get_order_book(graphname, cache):
    with open("{}/{}/order_book_{}.txt".format(DATA_DIR, graphname, cache)) as fp:
        lines = fp.readlines()
        offsets = []
        for l in lines:
            v = l.split(",")
            offsets.append([int(vv) for vv in v])
    return offsets 

def get_partition_offsets(graphname):
    with open("{}/{}/partition_offsets.txt".format(DATA_DIR, graphname)) as fp:
        lines = fp.readlines()
        assert(len(lines)== 1)
        offsets = [int(i) for i in lines[0].split(",")]
        return offsets
# Need a read function specifically for groot as
# graphs from from different sources have to preprocessed
# Remove fsize and testing
# Todo: Mostly we will use fixed fsize  
def get_process_graph(filename, fsize,  num_gpus, testing = False,):
    print("TODO: Refactor this not all function calls require the graph to be read")
    t1 = time.time()
    graphname = filename
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.int32)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.int32)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    if graphname not in synthetic_graphs and not graphname.startswith("synth"):
        assert(fsize == -1)
        results = read_meta_file(filename)
        fsize = results["feature_dim"]
        num_classes = results["num_classes"]
        if graphname == "mag240M":
            f_np = np.load("/home/ubuntu/data/mag240m_kddcup2021/processed/paper/node_feat.npy") 
            features = torch.from_numpy(f_np)
            print("Features read")
        else: 
            features = torch.from_numpy(np.fromfile(("{}/{}/features.bin").format(DATA_DIR,graphname)\
                                                            ,dtype = np.float32))
        features = features.reshape(num_nodes,fsize)
        labels = torch.from_numpy(\
                np.fromfile(("{}/{}/labels.bin".format(DATA_DIR, graphname)), dtype = np.intc)).to(\
                    torch.long)
        labels = labels.reshape(num_nodes,)
    else:
        #assert(fsize != -1)
        fsize = 400
        features = torch.rand(num_nodes,fsize)
        num_classes = 48
        labels = torch.randint(0,num_classes,(num_nodes,))
    print(num_nodes, num_edges, num_classes, "CLASS")

    assert(features.shape == (num_nodes,fsize))
    # features = torch.rand(num_nodes,fsize)
    indptr = indptr.astype(np.int32)
    indices = indices.astype(np.int32)
    dg_graph = dgl.graph(('csr',(indptr, indices, torch.arange(indices.shape[0], dtype = torch.int32))))
    print(dg_graph.edges()[0][:10])
    print("Using 32")
    # sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
    #     shape = (num_nodes,num_nodes))
    print("Scipy created")
    # dg_graph = dgl.from_scipy(sp)
    # dg_graph = dgl.to_homogeneous(dg_graph)
    dg_graph = dg_graph.astype(torch.int32)
    print("DG Graph Created")
    if graphname != "mag240M":
        # features = features.pin_memory()
        features = features.share_memory_()
    
    dg_graph.ndata["features"] = features
    dg_graph.ndata["labels"] = labels
    assert(dg_graph.ndata['features'].is_shared())
    idxs = ['train', 'val', 'test']
    ratio = {'train':.80,'val':.10,'test':.10}
    for idx in idxs:
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        if exists("{}/{}/{}_idx.bin".format(DATA_DIR, graphname, idx)):
            idx_mask = np.fromfile(
                "{}/{}/{}_idx.bin".format(DATA_DIR, graphname, idx), dtype=np.int32)
            mask[idx_mask] = True
        else:
            print("Generating random masks")
            if graphname == "mag240M" :
                continue
            mask  = torch.rand(num_nodes,) <  .8
          
        dg_graph.ndata["{}_mask".format(idx)] = mask
            
    
    partition_offsets = get_partition_offsets(graphname)
    
    print("All data created")
    t2 = time.time()
    print(f"Total time to read the graph {t2-t1}")
    # dg_graph = dg_graph.astype(torch.int32)
    return dg_graph, partition_offsets, num_classes


# Keep these untouched
def get_dgl_graph(name):
    dataset = DglNodePropPredDataset(name, root=DATA_DIR)
    graph, labels = dataset[0]
    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    features = graph.ndata['feat']
    graph.ndata['features']   = features
    assert features.shape[0] == num_nodes
    graph.ndata['labels'] = labels
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    idxs = ['train', 'valid', 'test']
    for idx in idxs:
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        mask[split_idx[idx]] = True
        graph.ndata[idx] = mask
    num_classes = int((max(labels[train_idx]) + 1).item())
    print("Num of classes", num_classes)
    return graph, None, num_classes, split_idx


# a,b = get_dgl_graph('ogbn-arxiv')
if __name__ == "__main__":
    # get_process_graph("amazon", -1)
    get_process_graph("ogbn-arxiv", -1, 4)
    get_process_graph("ogbn-products", -1, 4)
    get_process_graph("ogbn-papers100M", -1, 4)
    # get_process_graph("reordered-papers100M", -1)
    # get_process_graph("com-orkut", 128)
    print("Unit test get all process datasets !!! ")
