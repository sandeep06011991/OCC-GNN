
import torch
d = torch.device(0)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

a = torch.rand((100,100), device = d, requires_grad = True)
b = torch.rand((100,100), device = d, requires_grad = True)
c = torch.rand((100,100), device = d, requires_grad = True)
d = torch.rand((100,100), device = d, requires_grad = True)
import nvtx

with nvtx.annotate("Forward",color = 'blue'):
    a = a @ d
    with torch.cuda.stream(s1):
        x = a@b
    y = a@c
    torch.cuda.current_stream().wait_stream(s1)
    y = torch.sum(x + y)
with nvtx.annotate("Backward", color = "yellow"):
    y.backward()
print(d.grad)
