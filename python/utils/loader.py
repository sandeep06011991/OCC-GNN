import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.nodeproppred import NodePropPredDataset, DglNodePropPredDataset
import numpy as np
import csv
import heapq

dataset = DglNodePropPredDataset('ogbn-arxiv')
# graph, labels = dataset[0]
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0]

# get DGLgraph object to this point
# get the partition
partition = dgl.metis_partition(graph, 4, balance_edges=True)
with open('metadata', 'w') as f:
	for key in partition:
		# write to file len(partition[key].nodes())
		f.write(str(len(partition[key].nodes())) + ' ')

nodes = graph.nodes()
# function F, returns pid, ndegree
def F(node):
	for key in partition:
		if node in partition[key].ndata['_ID']:
			return int(key), -len(partition[key].in_edges(node, form='eid'))
	print('node not found')
	assert(False)

sortedNodes = list(heapq.merge(*[iter(nodes)], key=F))