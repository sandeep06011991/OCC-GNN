# Put in PaGraph/partition/
# Remove data import in both dg.py and ordering.py
#
import ordering
from dg import dg
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import os

ROOT_DIR = '/data/sandeep'
PAGRAPH_DIR = '/data/sandeep/pagraph/'
FILENAME = 'ogbn-products'
dataset = DglNodePropPredDataset(FILENAME, root=ROOT_DIR)
graph = dataset[0][0]
graph = graph.add_self_loop()
split_idx = dataset.get_idx_split()

scipyCOO = graph.adj(scipy_fmt='coo')
scipyCSR = scipyCOO.tocsc()

adj, vmap = ordering.reordering(scipyCSR, depth=3)

train_nids = split_idx['train']
train_nids = np.sort(vmap[train_nids])

p_v, p_trainv = dg(4, adj, train_nids, 3)

np.save(PAGRAPH_DIR+FILENAME + '_p_v', p_v)
np.save(PAGRAPH_DIR+FILENAME + '_p_trainv', p_trainv)

x = np.load(PAGRAPH_DIR+FILENAME + '_p_v'+'.npy', allow_pickle=True)
y = np.load(PAGRAPH_DIR+FILENAME + '_p_trainv'+'.npy', allow_pickle=True)
print(x, y)
