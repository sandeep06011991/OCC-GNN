import torch
ll = torch.nn.Linear(190,10).to(0)
r = torch.rand(29,190).to(0)
ll(r)
import test_cuslicer_consistency
test_cuslicer_consistency.csl3.getTensor()
