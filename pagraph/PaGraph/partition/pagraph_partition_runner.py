# Put in PaGraph/partition/
# Remove data import in both dg.py and ordering.py
#
from cProfile import label
from PaGraph.partition.ordering import reordering
from PaGraph.partition.dg import dg
from PaGraph.partition.utils import get_sub_graph
from PaGraph.data import get_labels, get_masks, get_labels
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import scipy.sparse as spsp
import os
import dgl
import torch

ROOT_DIR = '/data/sandeep'
PAGRAPH_DIR = '/data/sandeep/pagraph/arxiv/'
FILENAME = 'obgn_arxiv'
dataset = DglNodePropPredDataset('ogbn-arxiv', root=ROOT_DIR)
graph = dataset[0][0]
graph = graph.add_self_loop()
split_idx = dataset.get_idx_split()


def getMask(nodes, graph):
  res = torch.zeros(graph.nodes().shape[0], dtype=torch.bool)
  res[nodes] = True
  return res


masks = (getMask(split_idx['train'], graph), getMask(
    split_idx['valid'], graph), getMask(split_idx['test'], graph))

np.save(os.path.join(PAGRAPH_DIR, 'train.npy'), masks[0])
np.save(os.path.join(PAGRAPH_DIR, 'val.npy'), masks[1])
np.save(os.path.join(PAGRAPH_DIR, 'test.npy'), masks[2])

scipyCOO = graph.adj(scipy_fmt='coo')
scipyCSC = scipyCOO.tocsc()


# adj = spsp.load_npz(os.path.join(PAGRAPH_DIR, 'adj.npz'))
train_mask, val_mask, test_mask = get_masks(PAGRAPH_DIR)
train_nids = np.nonzero(train_mask)[0].astype(np.int64)
labels = dataset[0][1]


# ordering
print('re-ordering graphs...')
# adj = adj.tocsc()
adj, vmap = reordering(scipyCSC, depth=3)  # vmap: orig -> new
# np.save(PAGRAPH_DIR + 'adj', adj)
np.save(PAGRAPH_DIR + 'vmap', vmap)
# save to files
mapv = np.zeros(vmap.shape, dtype=np.int64)
mapv[vmap] = np.arange(vmap.shape[0])  # mapv: new -> orig
train_nids = np.sort(vmap[train_nids])
spsp.save_npz(os.path.join(PAGRAPH_DIR, 'adj.npz'), adj)
np.save(os.path.join(PAGRAPH_DIR, 'labels.npy'), labels[mapv])
np.save(os.path.join(PAGRAPH_DIR, 'train.npy'), train_mask[mapv])
np.save(os.path.join(PAGRAPH_DIR, 'val.npy'), val_mask[mapv])
np.save(os.path.join(PAGRAPH_DIR, 'test.npy'), test_mask[mapv])

labels = get_labels(PAGRAPH_DIR)

train_nids = split_idx['train']
train_nids = np.sort(vmap[train_nids])

p_v, p_trainv = dg(4, adj, train_nids, 3)

np.save(PAGRAPH_DIR + 'p_v', p_v)
np.save(PAGRAPH_DIR + 'p_trainv', p_trainv)

# x = np.load(PAGRAPH_DIR+FILENAME + '_p_v'+'.npy', allow_pickle=True)
# y = np.load(PAGRAPH_DIR+FILENAME + '_p_trainv'+'.npy', allow_pickle=True)
# print(x, y)

# save to file
partition_dataset = os.path.join(
    PAGRAPH_DIR, '{}naive'.format(4))
os.mkdir(partition_dataset)

dgl_g = dgl.DGLGraphStale(scipyCSC, readonly=True)
labels = get_labels(PAGRAPH_DIR)

for pid, (pv, ptrainv) in enumerate(zip(p_v, p_trainv)):
  print('generating subgraph# {}...'.format(pid))
  #subadj, sub2fullid, subtrainid = node2graph(adj, pv, ptrainv)
  subadj, sub2fullid, subtrainid = get_sub_graph(
      dgl_g, ptrainv, 3)
  sublabel = labels[sub2fullid[subtrainid]]
  # files
  subadj_file = os.path.join(
      partition_dataset,
      'subadj_{}.npz'.format(str(pid)))
  sub_trainid_file = os.path.join(
      partition_dataset,
      'sub_trainid_{}.npy'.format(str(pid)))
  sub_train2full_file = os.path.join(
      partition_dataset,
      'sub_train2fullid_{}.npy'.format(str(pid)))
  sub_label_file = os.path.join(
      partition_dataset,
      'sub_label_{}.npy'.format(str(pid)))
  spsp.save_npz(subadj_file, subadj)
  np.save(sub_trainid_file, subtrainid)
  np.save(sub_train2full_file, sub2fullid)
  np.save(sub_label_file, sublabel)
