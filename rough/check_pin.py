import os, psutil
def get_mem():
    process = psutil.Process()
    print(process.memory_info().rss/(1024 ** 3), "GB")


import torch

a = torch.ones(10 * 1024 ** 3) 

get_mem()

b = a.share_memory_()

get_mem()

c = b.pin_memory()

get_mem()

del b,a

get_mem()
