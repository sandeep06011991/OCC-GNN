import torch, dgl
import numpy as np
import os

from ogb.nodeproppred import DglNodePropPredDataset
import dgl.function as fn
from metis import *
from env import get_data_dir
from os.path import exists
ROOT_DIR = get_data_dir()
# Target dir is root dir + filename
TARGET_DIR = "{}/{}".format(ROOT_DIR, "papers100M")
os.makedirs(TARGET_DIR, exist_ok=True)

# if not exists(TARGET_DIR+'/cindptr.bin'):
if True:
     # File be run at one place for jupiter
    # Create binaries and partition file
    # Before any movement move all files to hardware.
    # Read from here and create pagraph partition
    # Read from here for quiver file
    name = "ogbn-papers100M"
    dataset = DglNodePropPredDataset(name, root=ROOT_DIR)
    
    graph, labels = dataset[0]
    features = graph.ndata['feat']
    split = dataset.get_idx_split()
    for k in split.keys():
        print(split[k].shape, k)
    graphs = dgl.metis_partition(graph, 4)    
    p_map = np.zeros(graph.num_nodes(), dtype = np.int32)
    for p in range(4):
        p_map[graphs[p].ndata['_ID']] = p
    
    train_idx = split['train']
    test_idx = split['test']
    edges = graph.edges()
    graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    num_nodes = graph.num_nodes()
    
    src,dest = graph.edges()


    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape), sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    # dgl graphs are in src to dest
    # indptr is for srcs
    # However sampling must start from dest
    # Thus reverse this.
    c_spmat = graph.adj(scipy_fmt = 'csr', transpose = True)
    c_indptr = c_spmat.indptr
    c_indices = c_spmat.indices

    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    # labels = labels[selected]
    nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
    labels = labels.flatten()
    labels[nan_lab] = 0
    # features = features[selected]
    print("Lables", torch.max(labels))
    # num_nodes = selected.shape[0]
    num_classes = int(torch.max(labels) + 1)
    assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    feature_dim = features.shape[1]

    # Check sum !
    csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()

    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)
    # if('train_idx' not in graph.ndata.keys()):
    #     split_idx = dataset.get_idx_split()
    
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
    with open(TARGET_DIR+'/test_idx.bin', 'wb') as fp:
        fp.write(test_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/partition_map_opt_4.bin', 'wb') as fp:
        fp.write(p_map.astype('int32').tobytes())
    csum_train = torch.sum(train_idx).item()
    # csum_test = torch.sum(val_idx).item()

    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    meta_structure["feature_dim"] = feature_dim
    meta_structure["csum_features"] = csum_features
    meta_structure["csum_labels"] = csum_labels
    meta_structure["csum_train"] = csum_train
    # meta_structure["csum_test"] = csum_test
    meta_structure["csum_offsets"] = csum_offsets
    meta_structure["csum_edges"] = csum_edges
    meta_structure["num_classes"] = num_classes
    with open(TARGET_DIR+'/meta.txt', 'w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k, meta_structure[k]))
    print("All data written!")

# import os
# username = os.environ['USER']
# if username == "spolisetty" :
#     generate_partition_file(ROOT_DIR , "reorder_papers100M")
