import torch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8,256)
        v = .01
        self.l1.weight = torch.nn.Parameter(torch.ones(256,8).fill_(v))
        self.l1.bias =  torch.nn.Parameter(torch.ones(256).fill_(v))

        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(256,2)
        self.l2.weight = torch.nn.Parameter(torch.ones(2,256).fill_(v))
        self.l2.bias = torch.nn.Parameter(torch.ones(2).fill_(v))

    def forward(self,inp):
        return self.l2(self.relu(self.l1(inp)))

model = Model()

optim = torch.optim.SGD(model.parameters(),lr = .01)
optim.zero_grad()
input = torch.rand(4,8)
loss = torch.nn.CrossEntropyLoss()
target = torch.empty((4,), dtype=torch.long).random_(2)
for i in range(100):
    # target = target.squeeze()
    output = model(input)
    output = loss(output, target)
    print("Loss !!",output)
    output.backward()
    optim.step()
# labels = torch.r  andint()
