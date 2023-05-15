import torch, dgl
import numpy as np
import os
import ogb 
import ogb.lsc

import dgl.function as fn
from metis import *
from env import get_data_dir
from os.path import exists
ROOT_DIR = get_data_dir()
# Target dir is root dir + filename
TARGET_DIR = "{}/{}".format(ROOT_DIR, "mag240M")
os.makedirs(TARGET_DIR, exist_ok=True)

if not exists(TARGET_DIR+'/cindptr.bin'):
     # File be run at one place for jupiter
    # Create binaries and partition file
    # Before any movement move all files to hardware.
    # Read from here and create pagraph partition
    # Read from here for quiver file
    #name = "ogbn-papers100M"
    dataset = ogb.lsc.mag240m.MAG240MDataset(root=ROOT_DIR)

    edge_index= dataset.edge_index('paper', 'paper')
    split = dataset.get_idx_split()
    print(split)
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes = dataset.num_papers)

    if not exists(TARGET_DIR + '/partition_map_opt_4.bin'):
        graphs = dgl.metis_partition(graph, 4)
        p_map = np.zeros(graph.num_nodes(), dtype = np.int32)
        for p in range(4):
            p_map[graphs[p].ndata['_ID']] = p
        with open(TARGET_DIR+'/partition_map_opt_4.bin', 'wb') as fp:
            fp.write(p_map.astype('int32').tobytes())
    else:
        print("Skip graph partition")

    labels = dataset.all_paper_label
    labels = torch.from_numpy(labels)

    num_nodes = dataset.num_papers


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
    print(graph.ndata.keys())
    if('train_idx' not in graph.ndata.keys()):
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test-dev']
        print("Train", train_idx.shape)
        print("Val", val_idx.shape)
        print("Test", test_idx.shape)

    labels = labels.flatten()
    for idx in [train_idx, test_idx, val_idx]:
        if torch.any(torch.isnan(labels[idx])):
            print(idx.shape , "Has NAN")
        if torch.any(labels[idx] < 0):
            print(idx.shape, "HASH neg")
    
    nan_lab = torch.where(torch.isnan(labels))
    nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
    neg_lab = torch.where(labels < 0)[0]
    
    labels = labels.flatten()
    labels[nan_lab] = 0
    labels[neg_lab] = 0

    # Skip preprocessing features vector 
    # torch.from_numpy and followed by ppinned and share
    # if not exists(TARGET_DIR + '/features.bin'):
    # print("Attempt to get features")
    # features = dataset.all_paper_feat
    # print("Feature Required Size ", features.shape, features.dtype)
    # features = torch.from_numpy(features)
    # assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    # feature_dim = features.shape[1]
    # with open(TARGET_DIR+'/features.bin', 'wb') as fp:
    # # This is an exception
    #     fp.write(features.numpy().astype('float16').tobytes())
    #     print("Features written")
    print("Lables", torch.max(labels))
    num_classes = int(torch.max(labels) + 1)
        
    # Check sum !
    # csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()

    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)
    # if('train_idx' not in graph.ndata.keys()):
    #     split_idx = dataset.get_idx_split()

    with open(TARGET_DIR+'/cindptr.bin', 'wb') as fp:
        fp.write(c_indptr.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/cindices.bin', 'wb') as fp:
        fp.write(c_indices.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/indptr.bin', 'wb') as fp:
        fp.write(indptr.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/indices.bin', 'wb') as fp:
        fp.write(indices.astype(np.int32).tobytes())
    
    with open(TARGET_DIR+'/labels.bin', 'wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/val_idx.bin', 'wb') as fp:
        fp.write(val_idx.astype('int32').tobytes())    
    with open(TARGET_DIR+'/test_idx.bin', 'wb') as fp:
        fp.write(test_idx.astype('int32').tobytes())
    with open(TARGET_DIR+'/train_idx.bin', 'wb') as fp:
        fp.write(train_idx.astype('int32').tobytes())
    #with open(TARGET_DIR+'/partition_map_opt_4.bin', 'wb') as fp:
    #    fp.write(p_map.astype('int32').tobytes())

    csum_train = np.sum(train_idx)
    csum_test = np.sum(val_idx)

    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    # Because this file is too big to process
    meta_structure["feature_dim"] = 768
    meta_structure["csum_features"] = 0
    meta_structure["csum_labels"] = csum_labels
    meta_structure["csum_train"] = csum_train
    # meta_structure["csum_test"] = csum_test
    meta_structure["csum_offsets"] = csum_offsets
    meta_structure["csum_edges"] = csum_edges
    meta_structure["num_classes"] = num_classes
    with open(TARGET_DIR+'/meta.txt', 'w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k, meta_structure[k]))
    print("All data written! to ", TARGET_DIR)

# import os
# username = os.environ['USER']
# if username == "spolisetty" :
#     generate_partition_file(ROOT_DIR , "reorder_papers100M")
