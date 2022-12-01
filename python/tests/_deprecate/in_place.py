import torch 
'''
f1 = torch.ones(10,10, requires_grad = True)
f = f1.clone()
x = torch.arange(10)
y = f * x
torch.sum(y).backward()
print(f1.grad)
print("Cross")
assert(False)
'''
f = torch.ones(10,10, requires_grad = True)
x = torch.arange(10, dtype = torch.float,  requires_grad = True)
y = f * x
y[5:6,:] = y[5:6,:] * 100 +  torch.ones(1,10, requires_grad = True)  * torch.ones(1,10, requires_grad = True)

torch.sum(y).backward()
print(f.grad)
print(x.grad)
print("Cross")

#Conclusion shuffle operator is same as normal operator"
