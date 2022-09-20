import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import os

ROOT_DIR = "/mnt/bigdata/sandeep"
TARGET_DIR = "/mnt/bigdata/sandeep"


def write_dataset_dataset(name, TARGET_DIR):
    dataset = DglNodePropPredDataset(name, root=ROOT_DIR)
    graph, labels = dataset[0]

    edges = graph.edges()
    graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    sparse_mat = graph.adj(scipy_fmt='csr')
    sparse_mat.sort_indices()
    assert(np.array_equal(np.ones(sparse_mat.data.shape), sparse_mat.data))
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    c_spmat = graph.adj(scipy_fmt = 'csr', transpose = False)
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
    print("All data written!")

# arg0 = dgl dataset name
# arg1 = full target directory
if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    # nname = ["ogbn-products","ogbn-arxiv"]
    nname = ["ogbn-papers100M"]
    for name in nname:
        target = TARGET_DIR + "/" + name
        os.makedirs(target, exist_ok=True)
        write_dataset_dataset(name, target)
