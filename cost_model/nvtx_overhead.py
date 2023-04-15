import torch.cuda.nvtx
import torch.cuda
e1 = torch.cuda.Event(enable_timing = True)
e2 = torch.cuda.Event(enable_timing = True)

for _ in range(10):
    e1.record()
    torch.cuda.nvtx.range_push("check")
    torch.cuda.nvtx.range_pop()
    e2.record()
    e2.synchronize()
    print("sync",e1.elapsed_time(e2)/1000)
