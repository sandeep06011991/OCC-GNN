# Use gdown
import sys
import dgl
import torch
import numpy as np
import subprocess
from env import get_data_dir
from os.path import exists
import scipy
import gdown
from sklearn.preprocessing import StandardScaler
import json
from metis import *
# All one time preprocessing goes here.
ROOT_DIR = get_data_dir()
TARGET_DIR = "{}/{}".format(ROOT_DIR, "amazon")



path_exists = exists("{}/adj_full.npz".format(TARGET_DIR))
# path_exists = False
if not path_exists:
    url = "https://drive.google.com/drive/folders/1uc76iCxBnd0ntNliosYDHHUc_ouXv9Iv"
    path = TARGET_DIR
    import os
    os.makedirs(path, exist_ok = True)
    gdown.download_folder(url = url, output = path)

# Read all datasets
prefix = "amazon"
adj_full = scipy.sparse.load_npz('{}/{}/adj_full.npz'.format(ROOT_DIR,prefix)).astype(np.bool)
adj_train = scipy.sparse.load_npz('{}/{}/adj_train.npz'.format(ROOT_DIR,prefix)).astype(np.bool)
role = json.load(open('{}/{}/role.json'.format(ROOT_DIR,prefix)))
feats = np.load('{}/{}/feats.npy'.format(ROOT_DIR,prefix))
class_map = json.load(open('{}/{}/class_map.json'.format(ROOT_DIR,prefix)))
class_map = {int(k):v for k,v in class_map.items()}
print(class_map[0])
num_classes = len(class_map[0])
assert len(class_map) == feats.shape[0]
# ---- normalize feats ----
train_nodes = np.array(list(set(adj_train.nonzero()[0])))
train_feats = feats[train_nodes]
scaler = StandardScaler()
scaler.fit(train_feats)
feats = scaler.transform(feats)
feats = torch.from_numpy(feats)

# Create all datastructures to process on jupiterself.
if not exists(TARGET_DIR+'/train_idx.bin'):
    graph = dgl.from_scipy(adj_train)
    classes = torch.zeros(graph.num_nodes(), num_classes,dtype = torch.long)
    classes[([i for i in class_map.keys()]),:] =torch.tensor([[i for i in class_map.values()]])
    classes = torch.max(classes,1)[1]
    labels = classes
    edges = graph.edges()
    graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape), sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
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
    features = feats
    assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    feature_dim = features.shape[1]
    csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()

    train_idx = torch.tensor(role['tr'])
    print("Train", train_idx.shape)

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

    csum_train = torch.sum(train_idx).item()

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

import os
username = os.environ['USER']
if username == "spolisetty" :
    print("start metis partitioning")
    generate_partition_file(ROOT_DIR , prefix)
    print("All data written!")
