import torch
from cslicer import stats
s = stats("ogbn-arxiv","occ",10)
print("Redundant computation in our partitioning")
print("Make partition map as a variable. ")
t = s.get_stats([0,1,2,3,4,5])
print(t.total_computation)
