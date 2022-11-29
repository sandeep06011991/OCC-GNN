import torch


# proof that my overlap technique pays off.
a = torch.rand(1000,1000, device = 0)
b =torch.rand(1000 ,1000, device = 0)
c = torch.rand(1000,1000, device = 0)
d = torch.rand(1000,1000)
e1 = torch.cuda.Event(enable_timing = True)
e2 = torch.cuda.Event(enable_timing = True)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
for i in range(100):
    e1.record()
    with torch.cuda.stream(s1):
        x = a + b
        e = x.to('cpu') + d
        e = e.to(0)
    with torch.cuda.stream(s2):
        y = a + c
    s1.synchronize()
    s2.synchronize()
    y = y + e
    e2.record()
    e2.synchronize()
    print("time", e1.elapsed_time(e2)/1000)


for i in range(100):
    e1.record()
    x = a + b
    e = x.to('cpu') + d
    y = a + c + e.to(0)
    e2.record()
    e2.synchronize()
    print("time", e1.elapsed_time(e2)/1000)
