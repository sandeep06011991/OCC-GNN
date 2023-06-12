import torch
from torchviz import make_dot

a = torch.rand(1000,1000, device = 0, requires_grad = True)
b = torch.rand(1000,1000, device = 0, requires_grad = True)


torch.cuda.nvtx.range_push("lnch overhead")
for _ in range(10):
    def f(a,b):
        c = a @ b
        return c.sum()
    d = f(a,b)
    make_dot(d, a, b)
            
torch.cuda.nvtx.range_pop()


torch.cuda.nvtx.range_push("clone overhead")
for _ in range(10):
    c = a @ b
    c.clone()
torch.cuda.nvtx.range_pop()


