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

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
    return DATA_DIR

DATA_DIR = get_data_dir()

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

def get_process_graph(filename, fsize):
    # DATA_DIR = "/data/sandeep"
    # graphname = "ogbn-products"
    graphname = filename
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.int64)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.int64)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    print(num_nodes, num_edges)
    if graphname not in synthetic_graphs and not graphname.startswith("synth"):
        assert(fsize == -1)
        results = read_meta_file(filename)
        fsize = results["feature_dim"]
        num_classes = results["num_classes"]
        features = torch.from_numpy(np.fromfile(("{}/{}/features.bin").format(DATA_DIR,graphname)\
                                                        ,dtype = np.float32))
        features = features.reshape(num_nodes,fsize)
        labels = torch.from_numpy(\
                np.fromfile(("{}/{}/labels.bin".format(DATA_DIR, graphname)), dtype = np.intc)).to(\
                    torch.long)
        labels = labels.reshape(num_nodes,)
    else:
        assert(fsize != -1)
        features = torch.rand(num_nodes,fsize)
        num_classes = 48
        labels = torch.randint(0,num_classes,(num_nodes,))

    assert(features.shape == (num_nodes,fsize))
    # features = torch.rand(num_nodes,fsize)
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
        shape = (num_nodes,num_nodes))
    dg_graph = dgl.from_scipy(sp)
    dg_graph = dgl.to_homogeneous(dg_graph)
    # features = features.pin_memory()
    dg_graph.ndata["features"] = features
    dg_graph.ndata["labels"] = labels

    idxs = ['train', 'val', 'test']
    ratio = {'train':.80,'val':.10,'test':.10}
    for idx in idxs:
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        if exists("{}/{}/{}_idx.bin".format(DATA_DIR, graphname, idx)):
            idx_mask = np.fromfile(
                "{}/{}/{}_idx.bin".format(DATA_DIR, graphname, idx), dtype=np.int64)
            mask[idx_mask] = True
        else:
            mask  = torch.rand(num_nodes,) <  .8
        dg_graph.ndata["{}_mask".format(idx)] = mask

    p_map = np.fromfile("{}/{}/partition_map_opt.bin".format(DATA_DIR,graphname),dtype = np.intc)
    # edges = dg_graph.edges()
    partition_map = torch.from_numpy(p_map)
    # assert(False)
    return dg_graph, partition_map, num_classes
    # , features, num_nodes, num_edges

# a,b = get_dgl_graph('ogbn-arxiv')
if __name__ == "__main__":
    # get_process_graph("amazon", -1)
    get_process_graph("ogbn-arxiv", -1)
    get_process_graph("ogbn-products", -1)
    # get_process_graph("reordered-papers100M", -1)
    # get_process_graph("com-orkut", 128)
    print("Unit test get all process datasets !!! ")
