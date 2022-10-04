import torch, dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import dgl.function as fn

ROOT_DIR = "/data/sandeep"
# Target dir is root dir + filename
TARGET_DIR = "/data/sandeep/reorder_papers100M"

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
train_idx = split['train']

edges = graph.edges()
graph.remove_edges(torch.where(edges[0] == edges[1])[0])
num_nodes = graph.num_nodes()
rev_graph = graph.reverse()
in_t = torch.zeros(num_nodes)
in_t[train_idx] = 1
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

src,dest = graph.edges()
selected_edges = torch.where(is_selected[src] & is_selected[dest])[0]
src_c = new_order[src[selected_edges]]
dest_c = new_order[dest[selected_edges]]
graph = dgl.graph((src_c,dest_c))
print("New dgl graph constructed")

sparse_mat = graph.adj(scipy_fmt='csr')
sparse_mat.sort_indices()
assert(np.array_equal(np.ones(sparse_mat.data.shape), sparse_mat.data))
indptr = sparse_mat.indptr
indices = sparse_mat.indices
c_spmat = graph.adj(scipy_fmt = 'csr', transpose = False)
c_indptr = c_spmat.indptr
c_indices = c_spmat.indices

num_edges = graph.num_edges()
num_nodes = graph.num_nodes()
labels = labels[selected]
nan_lab = torch.where(torch.isnan(labels.flatten()))[0]
labels = labels.flatten()
labels[nan_lab] = 0
features = features[selected]
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
train_idx =  new_order[train_idx]

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
