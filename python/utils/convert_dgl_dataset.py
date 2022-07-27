import sys
import dgl
import torch
import numpy as np
# All one time preprocessing goes here.
# ROOT_DIR = "/home/spolisetty/data"
# TARGET_DIR = "/home/spolisetty/data"
ROOT_DIR = "/data/sandeep"
TARGET_DIR = "/data/sandeep"
# TARGET_DIR = "/home/spolisetty/data/tests/gcn/"

def get_dataset(name):
    if name =="cora":
        graphs = dgl.data.CoraFullDataset()
        dataset = graphs[0]
        labels = dataset.ndata['label']
    if name =="pubmed":
        graphs = dgl.data.PubmedGraphDataset()
        dataset = graphs[0]
        labels = dataset.ndata['label']
    if name =="reddit":
        graphs = dgl.data.RedditDataset()
        dataset = graphs[0]
        labels = dataset.ndata['label']
    if name == "ogbn-arxiv":
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset,labels = DglNodePropPredDataset('ogbn-arxiv',root=ROOT_DIR)[0]
    if name == "ogbn-products":
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset,labels = DglNodePropPredDataset('ogbn-products',root=ROOT_DIR)[0]
    if name == "ogbn-papers100M":
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset,labels = DglNodePropPredDataset("ogbn-papers100M",root = ROOT_DIR)[0]
    if name == "ogbn-mag":
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset,labels = DglNodePropPredDataset('ogbn-mag',root = ROOT_DIR)[0]

    # Returns DGLHeteroGraph
    # Create dummy dataset for testing.
    return dataset,labels

def write_dataset_dataset(dataset,labels, TARGET_DIR):
    # dataset = dgl.add_self_loop(dataset)
    edges = dataset.edges()
    dataset.remove_edges(torch.where(edges[0]==edges[1])[0])
    sparse_mat = dataset.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape),sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    c_spmat = dataset.adj_sparse('csc')
    c_indptr = c_spmat[0]
    c_indices = c_spmat[1]
    print(indptr.shape)
    print(indices.shape)
    print("offset",indptr.sum())
    print("edges", indices.sum())
    num_edges = dataset.num_edges()
    num_nodes = dataset.num_nodes()
    nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
    labels = labels.flatten()
    labels[nan_lab] = 0
    print("Lables", torch.max(labels))
    num_classes = int(torch.max(labels) + 1)
    # dataset.num_classes
    features = dataset.ndata['feat']
    # labels = dataset[0].ndata['label']
    assert features.shape[0] == num_nodes
    assert labels.shape[0] == num_nodes
    feature_dim = features.shape[1]
    csum_features = torch.sum(features).item()
    csum_labels = torch.sum(labels).item()
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()
    # Partition the graph
    # p_id = dgl.metis_partition(dataset,4, balance_edges = True, mode = "recursive")
    # print("Done ")
    # p_map = torch.zeros((num_nodes))
    # L = 0
    # for k in p_id.keys():
    #     p_map[p_id[k].ndata['_ID']] = k
    #     L += (p_id[k].ndata['_ID']).shape[0]
    # # edges = dataset.edges()
    # cut = torch.sum(p_map[edges[0]] !=  p_map[edges[1]])
    # print("Current Edge Cut percentage.", cut, edges[0].shape[0], cut/edges[0].shape[0])
    # edge_ids = torch.where(p_map[edges[0]]!=p_map[edges[1]])[0]
    # # node_ids = torch.unique(torch.cat((edges[0][edge_ids],edges[1][edge_ids])))
    # node_ids = torch.unique(edges[0][edge_ids])
    # print("Halo nodes", node_ids.shape[0]/dataset.nodes().shape[0])
    # assert(L == num_nodes)
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
    with open(TARGET_DIR+'/cindptr.bin','wb') as fp:
        fp.write(c_indptr.numpy().astype(np.int64).tobytes())
    with open(TARGET_DIR+'/cindices.bin','wb') as fp:
        fp.write(c_indices.numpy().astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indptr.bin','wb') as fp:
        fp.write(indptr.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/indices.bin','wb') as fp:
        fp.write(indices.astype(np.int64).tobytes())
    with open(TARGET_DIR+'/features.bin','wb') as fp:
        fp.write(features.numpy().astype('float32').tobytes())
    with open(TARGET_DIR+'/labels.bin','wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/train_idx.bin','wb') as fp:
        fp.write(train_idx.numpy().astype('int64').tobytes())
    with open(TARGET_DIR+'/val_idx.bin','wb') as fp:
        fp.write(val_idx.numpy().astype('int64').tobytes())
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
    # nname = ["ogbn-products","ogbn-arxiv"]
    nname = ["ogbn-arxiv"]
    for name in nname:
        target = TARGET_DIR +"/" + name
        import os
        os.makedirs(target,exist_ok = True)
        dataset,labels = get_dataset(name)
        write_dataset_dataset(dataset,labels, target)
