import torch
import torch.multiprocessing as mp
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1 = nn.Linear(200,100)
        self.l2 = nn.Linear(200,200)

    def forward(self,b):
        return self.l2(self.l1(b))

def trainer_function(id,queues,barrier,model):
    print("start with training id",id)
    model = model.to(id)
    batch = queues[id].get(True)
    b1 = batch.to(id)
    del batch
    out = model(b1)
    out.sum().backward()
    # print(model.l2.weight.grad)
    # model.backward()
    barrier.wait()
    print("All done trainer")

def sampler_function(no_epochs,queues,barrier):
    print("start with no_epochs")
    rand = torch.rand(100,200).share_memory_()
    queues[0].put(rand[0:50])
    queues[1].put(rand[50:100])
    barrier.wait()
    print("All done")

    # a = torch.tensor([0,2]).share_memory_()



if __name__ =="__main__":
    n_gpus = 2
    no_epochs = 10
    processes = []
    queues = []
    model  = Model()

    barrier = mp.Barrier(n_gpus+1)
    for i in range(n_gpus):
        queues.append(mp.Queue())
    p = mp.Process(target = sampler_function, args = (no_epochs,queues,barrier))
    p.start()
    processes.append(p)
    for id in range(n_gpus):
        p = mp.Process(target = trainer_function, args = (id,queues,barrier,model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
