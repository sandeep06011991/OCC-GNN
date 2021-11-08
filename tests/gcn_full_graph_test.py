import dgl

import dgl
import torch
import dgl.function as fn
from convert_dgl_dataset import *

TARGET_DIR = "/home/spolisetty/data/tests/gcn"
name = "pubmed"
target = TARGET_DIR +"/pubmed"
# import os
# os.makedirs(target,exist_ok = True)
dataset = get_dataset(name)
graph = dataset[0]

class Model(torch.nn.Module):

    def __init__(self,dim1,dim2,dim3,graph):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim1,dim2)
        self.fc2 = torch.nn.Linear(dim2,dim3)
        self.relu1 = torch.nn.ReLU()
        # self.relu2 = torch.nn.ReLU()
        self.graph = graph
        # print(self.graph.ndata['feat'].shape,dim1,dim2)

    def forward(self):
        print("input feature sum : ", self.graph.ndata['feat'].sum())
        self.graph.update_all(fn.copy_u('feat', 'm'), fn.sum('m', 'f1'))
        print("first gcn aggr feature sum : ", self.graph.ndata['f1'].sum())
        ff1 = self.fc1(self.graph.ndata['f1'])
        print("first fc layer ", ff1.sum())
        self.graph.ndata['f2'] = self.relu1(ff1)
        print("first relu layer ", self.graph.ndata['f2'].sum())
        self.graph.update_all(fn.copy_u('f2', 'm'), fn.sum('m', 'f3'))
        print("second gcn aggr feature sum : ", self.graph.ndata['f3'].sum())
        self.graph.ndata['f4'] = (self.fc2(self.graph.ndata['f3']))
        print("second fc layer without", self.graph.ndata['f4'].sum())
        return self.graph.ndata['f4']
        pass

loss = torch.nn.CrossEntropyLoss()
model = Model(graph.ndata['feat'].shape[1],128,dataset.num_classes,graph)
optim = torch.optim.SGD(model.parameters(),lr = .1)
optim.zero_grad()
predicted = model.forward()
predicted.retain_grad()
labels = dataset[0].ndata['label']

# target = target.squeeze()
output = loss(predicted, labels)
print("Total loss is",output)
output.backward()
# optim.step()

# Goal perform forward and backward pass
with open(TARGET_DIR+'/W1.bin','wb') as fp:
    fp.write(model.fc1.weight.T.detach().flatten().numpy().astype('float32').tobytes())
if(model.fc1.bias != None):
    with open(TARGET_DIR+'/b1.bin','wb') as fp:
        fp.write(model.fc1.bias.detach().flatten().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/W2.bin','wb') as fp:
    fp.write(model.fc2.weight.T.detach().flatten().numpy().astype('float32').tobytes())
if(model.fc2.bias != None):
    with open(TARGET_DIR+'/b2.bin','wb') as fp:
        fp.write(model.fc2.bias.detach().flatten().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/predicted.bin','wb') as fp:
    fp.write(predicted.detach().flatten().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/dW1.bin','wb') as fp:
    fp.write(model.fc1.weight.grad.T.detach().flatten().numpy().astype('float32').tobytes())
if(model.fc1.bias != None):
    with open(TARGET_DIR+'/db1.bin','wb') as fp:
        fp.write(model.fc1.bias.grad.detach().flatten().numpy().astype('float32').tobytes())

# print(predicted.grad)
print("total loss", output.sum())
print("loss grad", predicted.grad.sum())
print(predicted[:3,:3])
print(labels[:3])
print(predicted.grad[:3,:3])
print("grad W1 FC1 sum",model.fc1.weight.grad.sum())
print("grad b1 FC1 sum",model.fc1.bias.grad.sum())
print("grad W2 FC2 sum",model.fc2.weight.grad.sum())
print("grad b2 FC2 sum",model.fc2.bias.grad.sum())
# total loss tensor(1.0985, grad_fn=<SumBackward0>)
# loss grad tensor(2.2992e-09)
# tensor([[ 0.1056,  0.2438,  0.0745],
#         [-0.0447,  0.0611,  0.0102],
#         [ 0.0168,  0.0495, -0.0148]], grad_fn=<SliceBackward>)
# tensor([1, 1, 0])
# tensor([[ 1.6268e-05, -3.2039e-05,  1.5770e-05],
#         [ 1.6008e-05, -3.2921e-05,  1.6913e-05],
#         [-3.3824e-05,  1.7456e-05,  1.6369e-05]])
# grad W1 FC1 sum tensor(5.8484)
# grad b1 FC1 sum tensor(0.3176)
# grad W2 FC2 sum tensor(-7.0926e-08)
# grad b2 FC2 sum tensor(0.)
# All ok!
#
# #

print("All ok!")
