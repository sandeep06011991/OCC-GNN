# Functions to build
# Matrix multiplication using cublasSge
import numpy
import torch

TARGET_DIR = "/mnt/homes/spolisetty/data/tests/matrix"


M = 1000
N = 156
K = 211
# M = 8
# N = 8
# K = 8
A = torch.rand((M,N),requires_grad = True)
B = torch.rand((N,K),requires_grad = True)

C = torch.matmul(A,B)
dA = torch.matmul(C,B.T)
dW = torch.matmul(A.T,C)
print(torch.sum(dW),"SUM")

with open(TARGET_DIR+'/A.bin','wb') as fp:
    fp.write(A.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/B.bin','wb') as fp:
    fp.write(B.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/C.bin','wb') as fp:
    fp.write(C.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/dA.bin','wb') as fp:
    fp.write(dA.detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/dW.bin','wb') as fp:
    fp.write(dW.detach().numpy().astype('float32').tobytes())

print("All data writing ok!")
