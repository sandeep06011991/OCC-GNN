import sys
import dgl
import torch
import numpy as np
# All one time preprocessing goes here.
ROOT_DIR = "/home/spolisetty/data"
TARGET_DIR = "/home/spolisetty/data"
# TARGET_DIR = "/home/spolisetty/data/tests/gcn/"

def get_dataset(name):
    if name =="cora":
        graphs = dgl.data.CoraFullDataset()
        dataset = graphs
    if name =="pubmed":
        graphs = dgl.data.PubmedGraphDataset()
        dataset = graphs
    if name =="reddit":
        graphs = dgl.data.RedditDataset()
        dataset = graphs
    # Returns DGLHeteroGraph
    # Create dummy dataset for testing.
    return dataset

def write_dataset_dataset(dataset, TARGET_DIR):
    sparse_mat = dataset[0].adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape),sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    print("offset",indptr.sum())
    print("edges", indices.sum())
    num_edges = dataset[0].num_edges()
    num_nodes = dataset[0].num_nodes()
    num_classes = dataset.num_classes
    features = dataset[0].ndata['feat']
    labels = dataset[0].ndata['label']
    assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    feature_dim = features.shape[1]
    csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()
    print(indptr.dtype)
    print(indptr[:10])

    assert indptr.shape == (num_nodes+1,)
    assert indices.shape == (num_edges,)
    if('train_idx' not in dataset[0].ndata.keys()):
        nodes = torch.randperm(num_nodes)
        train_idx = nodes[: int(len(nodes)*.8)]
        val_idx = nodes[int(len(nodes)*.8)+1:]
    # train_idx =  dataset[0].ndata['train_idx']
    # val_idx = dataset[0].ndata['val_idx']
    meta = {}
    with open(TARGET_DIR+'/indptr.bin','wb') as fp:
        fp.write(indptr.astype(np.intc).tobytes())
    with open(TARGET_DIR+'/indices.bin','wb') as fp:
        fp.write(indices.astype(np.intc).tobytes())
    with open(TARGET_DIR+'/features.bin','wb') as fp:
        fp.write(features.numpy().astype('float32').tobytes())
    with open(TARGET_DIR+'/labels.bin','wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/train_idx.bin','wb') as fp:
        fp.write(train_idx.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/val_idx.bin','wb') as fp:
        fp.write(val_idx.numpy().astype('int32').tobytes())
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
    with open(TARGET_DIR+'/meta.txt','w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k,meta_structure[k]))
    print("All data written!")
# arg0 = dgl dataset name
# arg1 = full target directory
if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    name = "reddit"
    target = TARGET_DIR +"/" + name
    import os
    os.makedirs(target,exist_ok = True)
    dataset = get_dataset(name)
    write_dataset_dataset(dataset, target)
