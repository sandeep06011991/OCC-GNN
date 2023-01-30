import torch
import numpy as np
import os
from os.path import exists
from metis import *

# Since this is the easiest dataset to generateself.
# Allow overwriting.
# file_exists = exists(path_to_file)

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

def write_synth_dataset(num_nodes, num_partitions , TARGET_DIR):
    # DGL graphs area always direction src to edges
    # dataset = DglNodePropPredDataset(name, root=ROOT_DIR)
    e1 = []
    e2 = []
    p_map = []
    for nd1 in range(num_nodes):
        p_map.append(nd1%num_partitions)
        for nd2 in range(num_nodes):
            if(nd1 != nd2) :
                    e1.append(nd1)
                    e2.append(nd2)
    graph = dgl.graph((torch.tensor(e1), torch.tensor(e2)))
    graph.ndata['feat'] = torch.ones(num_nodes,4)
    labels = torch.ones(num_nodes,)
    # graph, labels = dataset[0]
    # edges = graph.edges()
    # # dgl graph edges are always src to destination.
    # graph.remove_edges(torch.where(edges[0] == edges[1])[0])
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
    train_idx = test_idx = val_idx = torch.arange(num_nodes)

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
    print("Train", train_idx)
    print("Val", val_idx)
    print("Test", test_idx)
    # assert(Falscde)
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
    with open(TARGET_DIR+'/partition_map_opt.bin','wb') as fp:
        fp.write(torch.tensor(p_map).numpy().astype('int32').tobytes())
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
    # import os
    # username = os.environ['USER']
    # if username == "spolisetty" :
    #     generate_partition_file(ROOT_DIR, name)


# arg0 = dgl dataset name
# arg1 = full target directory
if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    clique_size = 8
    num_partitions = 2
    # Note papers 100M must be reordered
    name = "synth_{}_{}".format(clique_size, num_partitions)
    target = ROOT_DIR + "/" + name
    os.makedirs(target, exist_ok=True)
    write_synth_dataset(clique_size, num_partitions, target)
