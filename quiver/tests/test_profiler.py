import torch



for i in range(100):
    if i ==10:
        torch.cuda.profiler.start()
    a = torch.rand(100,100, device = 0)
    b = torch.rand(100, 100, device = 0)
    c = a @ b
    if i == 15:
        torch.cuda.profiler.stop()
