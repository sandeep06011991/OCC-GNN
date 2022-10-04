import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import os
from os.path import exists
from metis import *

file_exists = exists(path_to_file)

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

ROOT_DIR = get_data_dir()

def write_dataset_dataset(name, TARGET_DIR):
    # DGL graphs area always direction src to edges
    dataset = DglNodePropPredDataset(name, root=ROOT_DIR)
    graph, labels = dataset[0]
    edges = graph.edges()
    # dgl graph edges are always src to destination.
    graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape), sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    # c slicer should be processing from
    # dgl graphs are in src to dest
    # indptr is for srcs
    # However sampling must start from dest
    # Thus reverse this.
    c_spmat = graph.adj(scipy_fmt = 'csr', transpose = True)
    c_indptr = c_spmat.indptr
    c_indices = c_spmat.indices

    print(indptr.shape)
    print(indices.shape)
    print("offset", indptr.sum())
    print("edges", indices.sum())

    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
    labels = labels.flatten()
    labels[nan_lab] = 0

    print("Lables", torch.max(labels))

    num_classes = int(torch.max(labels) + 1)
    features = graph.ndata['feat']
    assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    feature_dim = features.shape[1]
    csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()

    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)
    if('train_idx' not in graph.ndata.keys()):
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
        print("Train", train_idx)
        print("Val", val_idx)
        print("Test", test_idx)

    # with open(TARGET_DIR + '/partition_map.bin','wb') as fp:
    #     fp.write(p_map.numpy().astype(np.int32).tobytes())
    with open(TARGET_DIR+'/cindptr.bin', 'wb') as fp:
        fp.write(c_indptr.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/cindices.bin', 'wb') as fp:
        fp.write(c_indices.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indptr.bin', 'wb') as fp:
        fp.write(indptr.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indices.bin', 'wb') as fp:
        fp.write(indices.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/features.bin', 'wb') as fp:
        fp.write(features.numpy().astype('float32').tobytes())
    with open(TARGET_DIR+'/labels.bin', 'wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/train_idx.bin', 'wb') as fp:
        fp.write(train_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/val_idx.bin', 'wb') as fp:
        fp.write(val_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/test_idx.bin', 'wb') as fp:
        fp.write(test_idx.numpy().astype('int64').tobytes())
    csum_train = torch.sum(train_idx).item()
    csum_test = torch.sum(val_idx).item()

    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    meta_structure["feature_dim"] = feature_dim
    meta_structure["csum_features"] = csum_features
    meta_structure["csum_labels"] = csum_labels
    meta_structure["csum_train"] = csum_train
    meta_structure["csum_test"] = csum_test
    meta_structure["csum_offsets"] = csum_offsets
    meta_structure["csum_edges"] = csum_edges
    meta_structure["num_classes"] = num_classes
    with open(TARGET_DIR+'/meta.txt', 'w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k, meta_structure[k]))
    import os
    username = os.environ['USER']
    if username == "spolisetty" :
        generate_partition_file(nname)


# arg0 = dgl dataset name
# arg1 = full target directory
if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    nname = ["ogbn-products","ogbn-arxiv"]
    # Note papers 100M must be reordered
    for name in nname:
        target = ROOT_DIR + "/" + name
        os.makedirs(target, exist_ok=True)
        write_dataset_dataset(name, target)
