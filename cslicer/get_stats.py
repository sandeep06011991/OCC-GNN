import torch
from cslicer import stats
s = stats("ogbn-arxiv","occ",10)
t = s.get_stats([0,1,2,3,4,5])
print(t.total_computation)
