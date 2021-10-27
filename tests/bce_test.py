
TARGET_DIR = "/mnt/homes/spolisetty/data/tests/bce"

import torch
N = 19717
M = 3
N = 1024
M = 5

input = torch.ones((N,M),requires_grad = True) *20
input.retain_grad()
loss = torch.nn.CrossEntropyLoss()
target = torch.empty((N,), dtype=torch.long).random_(M)
# target = target.squeeze()

output = loss(input, target)
print("Loss !!",output)
output.backward()
# print(target)

with open(TARGET_DIR+'/input.bin','wb') as fp:
    fp.write(input.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/target.bin','wb') as fp:
    fp.write(target.numpy().astype('int32').tobytes())
with open(TARGET_DIR+'/out.bin','wb') as fp:
    fp.write(output.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/grad.bin','wb') as fp:
    fp.write(input.grad.numpy().astype('float32').tobytes())
