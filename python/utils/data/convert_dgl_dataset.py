import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import os
from os.path import exists
from env import *
import dgl
import time 

# Not using logging as I dont need to persist the time taken to preprocess.
# When that need arises move all prints to logging 
ROOT_DIR = get_data_dir()

def write_dataset_dataset(name, TARGET_DIR, num_partitions = 4):
    # DGL graphs area always direction src to edges
    print("Get DGL Graph", name, TARGET_DIR)
    t1 = time.time()
    dataset = DglNodePropPredDataset(name, root=ROOT_DIR)
    graph, labels = dataset[0]
    edges = graph.edges()
    graph.remove_edges(torch.where(edges[0] == edges[1])[0])
    edges = graph.edges()
    t2 = time.time()
    print(f"Time to read graph:{t2-t1}")
    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    assert(num_edges == edges[0].shape[0])
    features = graph.ndata['feat']
    assert features.shape[0] == num_nodes

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    
    print("Train", train_idx.shape)
    print("Val", val_idx.shape)
    print("Test", test_idx.shape)


    if False:
        # Always back this up
        print("Note check if any new library functions")
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        mask[train_idx] = 1
        graphs = dgl.metis_partition(graph, num_partitions, balance_ntypes = mask)    
        p_map = np.zeros(graph.num_nodes(), dtype = np.int32)
        for p in range(num_partitions):
            p_map[graphs[p].ndata['_ID']] = p
            print(f"Nodes in partition {graphs[p].ndata['_ID'].shape}")
        with open(TARGET_DIR + f'/partition_map_opt_{num_partitions}.bin','wb') as fp:
            fp.write(p_map.astype(np.int32).tobytes())
        partition_map = torch.from_numpy(p_map)
    else: 
        p_map = (np.fromfile(TARGET_DIR + f'/partition_map_opt_{num_partitions}.bin', dtype = np.int32))
        partition_map = torch.from_numpy(p_map)

    new_order = None
    degrees = graph.out_degrees()
    ordered_partition_nodes = []
    partition_offsets = [0]
    for i in range(num_partitions):
        nodes_in_partition = torch.where(partition_map == i)[0]
        _, indices = torch.sort(degrees[nodes_in_partition], descending = True)
        ordered_partition_nodes.append(nodes_in_partition[indices])
        partition_offsets.append(partition_offsets[-1] + indices.shape[0])
    with open(TARGET_DIR+'/partition_offsets.txt', 'w') as fp:
        fp.write(",".join([str(i) for i in partition_offsets]))

    new_to_old_order = torch.cat(ordered_partition_nodes, dim = 0)    
    values,old_to_new_order = torch.sort(new_to_old_order)
    assert(torch.all(torch.arange(graph.num_nodes()) == values))

    # Start reordering everything
    graph = dgl.graph((old_to_new_order[edges[0]], old_to_new_order[edges[1]]), num_nodes = num_nodes)
    sparse_mat = graph.adj()
    # Storing the graph in CSC format is ideal for sampling 
    # DGL Graph can be constructed from CSC format
    indptr, indices, _ = sparse_mat.csc()
    out_degrees = graph.out_degrees()
    
    with open(TARGET_DIR + '/out_degrees.bin', 'wb') as fp:
        fp.write(out_degrees.numpy().astype(np.int32).tobytes())
    
    assert torch.all(graph.in_degrees() == (indptr[1:] - indptr[:-1]))
    assert indptr.shape == (num_nodes+1,)
    print(indices.shape, num_edges, "Error")
    assert indices.shape == (num_edges,)
    csum_offsets = indptr.sum()
    csum_edges = indices.sum()

    with open(TARGET_DIR+'/indptr.bin', 'wb') as fp:
        fp.write(indptr.numpy().astype(np.int32).tobytes())
    with open(TARGET_DIR+'/indices.bin', 'wb') as fp:
        fp.write(indices.numpy().astype(np.int32).tobytes())
    features = features[new_to_old_order]
    with open(TARGET_DIR+'/features.bin', 'wb') as fp:
        fp.write(features.numpy().astype('float32').tobytes())
    with open(TARGET_DIR + '/new_to_old_order.bin','wb') as fp:
        fp.write(new_to_old_order.numpy().astype('int32').tobytes())
    with open(TARGET_DIR + '/old_to_new_order.bin','wb') as fp:
        fp.write(old_to_new_order.numpy().astype('int32').tobytes())
    labels = labels.flatten()
    assert(not torch.any(torch.isnan(labels[train_idx])))
    labels = labels[new_to_old_order]
    
    with open(TARGET_DIR+'/labels.bin', 'wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    
    assert features.shape[0] == num_nodes
    feature_dim = features.shape[1]
    csum_features = torch.sum(features).item()
    
    train_idx = old_to_new_order[train_idx]
    test_idx = old_to_new_order[test_idx]
    val_idx = old_to_new_order[val_idx]
    
    assert(not torch.any(torch.isnan(labels[train_idx])))
    assert(not torch.any(labels[train_idx] < 0))
    for idx in [train_idx, test_idx, val_idx]:
        nan_lab = torch.where(torch.isnan(labels))
        labels[nan_lab] = 0
        assert labels.shape[0] == num_nodes
    csum_labels = torch.sum(labels).item()
    num_classes = int(torch.max(labels) + 1)


    with open(TARGET_DIR+'/train_idx.bin', 'wb') as fp:
        fp.write(train_idx.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/val_idx.bin', 'wb') as fp:
        fp.write(val_idx.numpy().astype('int32').tobytes())
    with open(TARGET_DIR+'/test_idx.bin', 'wb') as fp:
        fp.write(test_idx.numpy().astype('int32').tobytes())
    csum_train = torch.sum(train_idx).item()
    csum_test = torch.sum(val_idx).item()
    csum_val = torch.sum(test_idx).item()
    meta_structure = {}
    meta_structure["num_nodes"] = num_nodes
    meta_structure["num_edges"] = num_edges
    meta_structure["csum_features"] = int(csum_features)
    meta_structure["feature_dim"] = feature_dim
    meta_structure["csum_labels"] = csum_labels
    meta_structure["csum_train"] = csum_train
    meta_structure["csum_test"] = csum_test
    meta_structure["csum_offsets"] = csum_offsets
    meta_structure["csum_edges"] = csum_edges
    meta_structure["num_classes"] = num_classes
    meta_structure["fbytes"] = 32 
    print(meta_structure)
    with open(TARGET_DIR+'/meta.txt', 'w') as fp:
        for k in meta_structure.keys():
            fp.write("{}={}\n".format(k, meta_structure[k]))


# arg0 = dgl dataset name
# arg1 = full target directory
if __name__=="__main__":
    # assert(len(sys.argv) == 3)
    nname = ["ogbn-papers100M"]
    nname = [ "ogbn-papers100M"]
    # Note papers 100M must be reordered
    for name in nname:
        target = ROOT_DIR + "/" + name
        os.makedirs(target, exist_ok=True)
        write_dataset_dataset(name, target)
