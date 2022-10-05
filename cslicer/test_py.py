# How to use cslicer in pythonic way.

import torch
from cslicer import cslicer
graphname = "ogbn-arxiv"
number_of_epochs = 1
minibatch_size =4096
num_nodes = 169343
# // Get this from data
# // All absent
storage_map = [[],[],[],[]]
graphnames = ["ogbn-arxiv","ogbn-products"]
for graphname in graphnames:
    csl = cslicer(graphname, storage_map, 10, False)
#in_nodes = [0,1,2,3,4,5,6]
#csl.getSample(in_nodes)
#print("All success !")

# from cslicer import stats
s = stats("ogbn-arxiv","occ",10)
t = s.get_stats([0,1,2,3,4,5])
print(t)
