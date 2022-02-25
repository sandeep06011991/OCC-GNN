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


def get_dgl_graph(filename):
    DATA_DIR = "/home/spolisetty/data"
    # graphname = "ogbn-products"
    graphname = filename
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.intc)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.intc)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    fsize = 1024
    features = torch.rand(num_nodes,fsize)
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
        shape = (num_nodes,num_nodes))
    dg_graph = DGLGraph(sp)
    dg_graph = dgl.to_homogeneous(dg_graph)
    dg_graph.ndata["features"] = features
    num_classes = 40
    dg_graph.ndata["labels"] = torch.randint(0,40,(num_nodes,))
    a = torch.rand((num_nodes,))
    dg_graph.ndata["train_mask"] = a < .80
    dg_graph.ndata["val_mask"] = (a >= .80) & (a <.90)
    dg_graph.ndata["test_mask"] = a >=.90
    p_map = np.fromfile("{}/{}/partition_map.bin".format(DATA_DIR,graphname),dtype = np.intc)
    # edges = dg_graph.edges()
    partition_map = torch.from_numpy(p_map)
    return dg_graph, partition_map
    # , features, num_nodes, num_edges

# a,b = get_dgl_graph('ogbn-arxiv')
