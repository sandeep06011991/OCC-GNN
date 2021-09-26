import torch
TARGET_DIR = "/mnt/homes/spolisetty/data/tests/linear"

import torch
N = 1024
M = 40
K = 16
# N = 4
# M = 4
# K = 12

input = torch.rand((N,M),requires_grad = True)
layer1 = torch.nn.Linear(M, K)
print(layer1.weight.shape)
# assert(False)
layer2 = torch.nn.ReLU()

out1 = layer1(input)
out1.retain_grad()
output = layer2(out1)
output.retain_grad()
output.sum().backward()

### Write all inputs
with open(TARGET_DIR+'/input.bin','wb') as fp:
    fp.write(input.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/W.bin','wb') as fp:
    fp.write(layer1.weight.T.detach().numpy().astype('float32').tobytes())
if layer1.bias!=None:
    assert(layer1.bias.shape == (K,))
    with open(TARGET_DIR+'/b.bin','wb') as fp:
        fp.write(layer1.bias.detach().numpy().astype('float32').tobytes())
### Write intermediate outputs
with open(TARGET_DIR+'/out1.bin','wb') as fp:
    fp.write(out1.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/out2.bin','wb') as fp:
    fp.write(output.detach().numpy().astype('float32').tobytes())
### Write intermediate gradients
with open(TARGET_DIR+'/dW.bin','wb') as fp:
    fp.write(layer1.weight.grad.T.numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/dX.bin','wb') as fp:
    fp.write(input.grad.numpy().astype('float32').tobytes())
print(input.grad)
if layer1.bias!=None:
    with open(TARGET_DIR+'/db.bin','wb') as fp:
        fp.write(layer1.bias.grad.numpy().astype('float32').tobytes())

print("All ok !!")
