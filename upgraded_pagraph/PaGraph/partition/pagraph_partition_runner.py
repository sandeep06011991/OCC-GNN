# Put in PaGraph/partition/
# Remove data import in both dg.py and ordering.py
# Run this file on jupiter as we this code is written with latest dgl and ogb version

from cProfile import label
from PaGraph.partition.ordering import reordering
from PaGraph.partition.dg import dg
from PaGraph.partition.utils import get_sub_graph
from PaGraph.data import get_labels, get_masks, get_labels
import scipy
#from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import scipy.sparse as spsp
import os
import dgl
import torch
import time

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
        ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/pagraph"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
        ROOT_DIR = "/home/spolisetty/OCC-GNN/pagraph"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
    return DATA_DIR,ROOT_DIR

DATA_DIR,ROOT_DIR = get_data_dir()
import logging


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

def get_process_graph(filename):
    # DATA_DIR = "/data/sandeep"
    # graphname = "ogbn-products"
    os.makedirs('{}/logs'.format(ROOT_DIR),exist_ok = True)
    FILENAME= ('{}/logs/partition_{}.txt'.format(ROOT_DIR, filename))
    fileh = logging.FileHandler(FILENAME, 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.addHandler(fileh)      # set the new handler
    log.setLevel(logging.INFO)
    graphname = filename
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.int64)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.int64)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    print(num_nodes, num_edges)
    # features = torch.rand(num_nodes,fsize)
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
        shape = (num_nodes,num_nodes))
    dg_graph = dgl.from_scipy(sp)
    # assert(False)
    print(num_nodes, num_edges)
    synthetic_graphs = ["com-orkut"]
    if graphname not in synthetic_graphs :
        train_idx = np.fromfile('{}/{}/train_idx.bin'.format(DATA_DIR, graphname), dtype = np.int64)
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
        train_idx = torch.from_numpy(train_idx)
    else:
        fsize = 512
        train_idx = torch.arange(num_nodes)
        features = torch.rand(num_nodes,fsize)
        num_classes = 48
        labels = torch.randint(0,num_classes,(num_nodes,))
        assert(features.shape == (num_nodes,fsize))
        # features = torch.rand(num_nodes,fsize)
        # features = features.pin_memory()
    dg_graph.ndata["features"] = features
    dg_graph.ndata["labels"] = labels
    return dg_graph, train_idx

def getMask(nodes, graph):
  res = torch.zeros(graph.nodes().shape[0], dtype=torch.bool)
  res[nodes] = True
  return res

# ROOT_DIR = '/data/sandeep'
def pagraph_partition(FILENAME):
    dg_graph, train_idx = get_process_graph(FILENAME)
    PAGRAPH_DIR = '{}/pagraph/{}/'.format(DATA_DIR, FILENAME)
    os.makedirs(PAGRAPH_DIR,exist_ok = True)
    graph = dg_graph
    features = graph.ndata['features']
    labels = graph.ndata['labels']
    np.save(os.path.join(PAGRAPH_DIR,'feat.npy'),features)

    train_mask = (getMask(train_idx, graph))
    # , getMask(split_idx['valid'], graph), getMask(split_idx['test'], graph))

    np.save(os.path.join(PAGRAPH_DIR, 'train.npy'), train_mask)
    # np.save(os.path.join(PAGRAPH_DIR, 'val.npy'), masks[1])
    # np.save(os.path.join(PAGRAPH_DIR, 'test.npy'), masks[2])

    graph = dg_graph
    scipyCOO = graph.adj(scipy_fmt='coo')
    scipyCSC = scipyCOO.tocsc()

    # adj = spsp.load_npz(os.path.join(PAGRAPH_DIR, 'adj.npz'))
    train_nids = train_idx.numpy()
    log = logging.getLogger()
    # ordering
    print('re-ordering graphs...')
    print('skip reordering')
    adj = scipyCSC
    # adj = adj.tocsc()
    print("Skipping reordering")
    #adj, vmap = reordering(scipyCSC, depth=3)  # vmap: orig -> new
    vmap = torch.arange(graph.num_nodes())
    # np.save(PAGRAPH_DIR + 'adj', adj)
    vmap = torch.arange(graph.num_nodes())
    np.save(PAGRAPH_DIR + 'vmap', vmap)
    # save to files
    mapv = np.zeros(vmap.shape, dtype=np.int64)
    mapv[vmap] = np.arange(vmap.shape[0])  # mapv: new -> orig
    train_nids = np.sort(vmap[train_nids])
    spsp.save_npz(os.path.join(PAGRAPH_DIR, 'adj.npz'), adj)
    np.save(os.path.join(PAGRAPH_DIR, 'labels.npy'), labels[mapv])
    np.save(os.path.join(PAGRAPH_DIR, 'train.npy'), train_mask[mapv])
    # np.save(os.path.join(PAGRAPH_DIR, 'val.npy'), val_mask[mapv])
    # np.save(os.path.join(PAGRAPH_DIR, 'test.npy'), test_mask[mapv])

    s = time.time()
    log.info("start partitioning")
    p_v, p_trainv = dg(4, adj, train_nids, 3)
    e = time.time()

    print("Total partitioning time", e-s)
    np.save(PAGRAPH_DIR + 'p_v', p_v)
    np.save(PAGRAPH_DIR + 'p_trainv', p_trainv)

# x = np.load(PAGRAPH_DIR+FILENAME + '_p_v'+'.npy', allow_pickle=True)
# y = np.load(PAGRAPH_DIR+FILENAME + '_p_trainv'+'.npy', allow_pickle=True)
# print(x, y)

    # save to file
    partition_dataset = os.path.join(
        PAGRAPH_DIR, '{}naive'.format(4))
    if not os.path.exists(partition_dataset):
        os.mkdir(partition_dataset)
    dgl_g = dgl.DGLGraphStale(scipyCSC, readonly=True)

    for pid, (pv, ptrainv) in enumerate(zip(p_v, p_trainv)):
        print('generating subgraph# {}...'.format(pid))
        #subadj, sub2fullid, subtrainid = node2graph(adj, pv, ptrainv)
        subadj, sub2fullid, subtrainid = get_sub_graph( \
                                                    dgl_g, ptrainv, 3)
        sublabel = labels[sub2fullid[subtrainid]]
        # files
        subadj_file = os.path.join(
                            partition_dataset,
                                'subadj_{}.npz'.format(str(pid)))
        sub_trainid_file = os.path.join(
        partition_dataset,
        'sub_trainid_{}.npy'.format(str(pid)))
        sub_train2full_file = os.path.join(
        partition_dataset,
        'sub_train2fullid_{}.npy'.format(str(pid)))
        sub_label_file = os.path.join(
        partition_dataset,
        'sub_label_{}.npy'.format(str(pid)))
        spsp.save_npz(subadj_file, subadj)
        np.save(sub_trainid_file, subtrainid)
        np.save(sub_train2full_file, sub2fullid)
        np.save(sub_label_file, sublabel)

if __name__=="__main__":
    graph_names = ["ogbn-arxiv","ogbn-products", "reorder-papers100M", "amazon", "com-orkut"]
    print("others are done!")
    graph_names = ["amazon"]
    for graph in graph_names:
        pagraph_partition(graph)
