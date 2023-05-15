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
def mode(is_test):
    if is_test:
        TARGET_DIR = "{}/test_{}".format(ROOT_DIR, "reorder_mag240M")
    else:
        TARGET_DIR = "{}/train_{}".format(ROOT_DIR, "reorder_mag240M")
    os.makedirs(TARGET_DIR, exist_ok=True)

     # File be run at one place for jupiter
    # Create binaries and partition file
    # Before any movement move all files to hardware.
    # Read from here and create pagraph partition
    # Read from here for quiver file
    #name = "ogbn-papers100M"
    dataset = ogb.lsc.mag240m.MAG240MDataset(root=ROOT_DIR)

    edge_index= dataset.edge_index('paper', 'paper')
    features = dataset.paper_feat
    split = dataset.get_idx_split()
    for k in split.keys():
        print(split[k].shape, k)
    if not is_test:
        idx = split['train']
    else:
        idx = split['test-dev' ]
    labels = dataset.paper_label
    num_nodes = dataset.num_papers
    print("Num of nodes", num_nodes)
    dg_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes = num_nodes)
    print(dg_graph)
    rev_graph = dg_graph.reverse()
    in_t = torch.zeros(num_nodes)
    in_t[idx] = 1
    in_t = in_t.reshape(num_nodes,1)
    rev_graph.ndata['in'] = in_t
    total = in_t.clone()
    num_hops  = 3

    for i in range(num_hops):
        rev_graph.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
        rev_graph.ndata['in'] = rev_graph.ndata['out']
        total += rev_graph.ndata['out']
        # Get all nodes that will be touched

    print("selected vertices", torch.where(total !=0)[0].shape, "total", num_nodes)
    selected = torch.where(total != 0)[0]
    is_selected = torch.zeros(num_nodes, dtype = torch.bool)
    new_order = torch.zeros(num_nodes, dtype = torch.long)
    is_selected[selected] = True
    new_order[selected] = torch.arange(selected.shape[0],dtype = torch.long)
    print("reorder complete !")
    # Reorder all nodes that have been touched with new orders

    src,dest = edge_index
    selected_edges = torch.where(is_selected[src] & is_selected[dest])[0]
    src_c = new_order[src[selected_edges]]
    dest_c = new_order[dest[selected_edges]]
    graph = dgl.graph((src_c,dest_c))
    print("New dgl graph constructed")
    if not exists(TARGET_DIR+'/partition_map_opt_4.bin'):
        graphs = dgl.metis_partition(graph, 4)
        p_map = np.zeros(graph.num_nodes(), dtype = np.int32)
        for p in range(4):
            p_map[graphs[p].ndata['_ID']] = p
        with open(TARGET_DIR+'/partition_map_opt_4.bin', 'wb') as fp:
                fp.write(p_map.astype('int32').tobytes())
 
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
    labels = labels[selected]
    labels = torch.from_numpy(labels)
    nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
    labels = labels.flatten()
    labels[nan_lab] = 0
    features = features[selected]
    print("Feature Required Size ", features.shape)
    features = torch.from_numpy(features)
    print("Lables", torch.max(labels))
    num_nodes = selected.shape[0]
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
    idx =  new_order[idx]

    with open(TARGET_DIR+'/cindptr.bin', 'wb') as fp:
        fp.write(c_indptr.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/cindices.bin', 'wb') as fp:
        fp.write(c_indices.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/indptr.bin', 'wb') as fp:
        fp.write(indptr.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/indices.bin', 'wb') as fp:
        fp.write(indices.astype(np.int32).tobytes())
    with open(TARGET_DIR+'/features.bin', 'wb') as fp:
        fp.write(features.numpy().astype('float32').tobytes())
    with open(TARGET_DIR+'/labels.bin', 'wb') as fp:
        fp.write(labels.numpy().astype('int32').tobytes())
    if is_test:
        csum_train = 0
        csum_test = torch.sum(idx).item()
        with open(TARGET_DIR+'/test_idx.bin', 'wb') as fp:
            fp.write(idx.numpy().astype('int32').tobytes())
    else:
        csum_test = 0
        csum_train = torch.sum(idx).item()
        with open(TARGET_DIR+'/train_idx.bin', 'wb') as fp:
            fp.write(idx.numpy().astype('int32').tobytes())

    

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
    print("All data written!", mode)

# import os
# username = os.environ['USER']
# if username == "spolisetty" :
#     generate_partition_file(ROOT_DIR , "reorder_papers100M")
mode(True)
mode(False)