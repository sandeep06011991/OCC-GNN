# How to use cslicer in pythonic way.

import torch
from cslicer import cslicer
graphname = "ogbn-arxiv"
number_of_epochs = 1
minibatch_size =4096
num_nodes = 169343
# // Get this from data
# // All absent
storage_map = [[0],[1],[2],[3]]
csl = cslicer(graphname, storage_map, 10, False)
in_nodes = [0,1,2,3,4,5,6]
csl.getSample(in_nodes)
print("All success !")
