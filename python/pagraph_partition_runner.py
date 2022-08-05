# Put in PaGraph/partition/
# Remove data import in both dg.py and ordering.py
#

import ordering
from dg import dg
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np

ROOT_DIR = '/data/sandeep'
PAGRAPH_DIR = '/data/sandeep/pagraph'
dataset = DglNodePropPredDataset('ogbn-products', root=ROOT_DIR)
graph = dataset[0][0]
graph = graph.add_self_loop()
split_idx = dataset.get_idx_split()

scipyCOO = graph.adj(scipy_fmt='coo')
scipyCSR = scipyCOO.tocsc()

adj, vmap = ordering.reordering(scipyCSR, depth=3)

train_nids = split_idx['train']
train_nids = np.sort(vmap[train_nids])

p_v, p_trainv = dg(4, adj, train_nids, 3)

all_nodes = []
for i in p_v:
  all_nodes.extend(i.tolist())
all_nodes = np.unique(all_nodes)

cache_rate = [len(i)/len(all_nodes) for i in p_v]
cache_rate
