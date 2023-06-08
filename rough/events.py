import torch 


a = torch.rand(10000,10000).to(0)
b = torch.rand(10000,10000).to(0)

c= torch.rand(10000,10000).to(1)
d = torch.rand(10000,10000).to(1)

e1 = torch.cuda.Event(enable_timing = True)



for i in range(100):
    torch.cuda.set_device(0)
    f = a@b
    e1.record()

    torch.cuda.set_device(1)
    e1.wait()
    m = c@ d

torch.cuda.synchronize()
print("done")
