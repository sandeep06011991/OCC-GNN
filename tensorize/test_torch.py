import torch, torch_tensorize
import tensorize
import time 
for ii in range(5):
    b = torch.tensor([100]).to(device = 1)
    t1 = time.time()
    a = torch_tensorize.getlist()
    t11 = time.time()
    a.to(device = 0)

    t2 = time.time()
    a = tensorize.test_list()
    print(a[:10])
    b = torch.tensor(a,device = 1)
    t3 = time.time()
    print("directly tensoirzed", t11 - t1)
    print("data movememnt", t2 - t11)
    print("indirectly tensorized", t3 - t2)
