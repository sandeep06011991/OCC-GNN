import dgl
import torch.nn as nn
import torch
from torch.nn.parallel import gather
from dgl.nn.pytorch.conv.gatconv import GATConv
import dgl.nn.pytorch.conv.sageconv as sgc
import torch.multiprocessing as mp
from data.test_bipartite import get_dummy_bipartite_graph
from torch.nn.parallel import DistributedDataParallel
from layers.dist_gatconv import DistGATConv


def test_base():
    src_ids = []
    dest_ids = []
    for dest in range(4):
        for source in range(8):
            src_ids.append(source)
            dest_ids.append(dest)

    g = dgl.create_block((src_ids, dest_ids), 8, 4)
    dglGat = GATConv((4, 4), 8, num_heads=1)

    dglGat.fc_src.weight = torch.nn.Parameter(
        torch.ones(dglGat.fc_src.weight.shape))
    dglGat.fc_dst.weight = torch.nn.Parameter(
        torch.ones(dglGat.fc_dst.weight.shape))
    dglGat.attn_l = torch.nn.Parameter(
        torch.ones(dglGat.attn_l.shape))
    dglGat.attn_r = torch.nn.Parameter(
        torch.ones(dglGat.attn_r.shape))

    v_ones = torch.ones(4, 4)
    u_ones = torch.ones(8, 4)

    res = dglGat(g, (u_ones, v_ones), get_attention=False)
    forward_correct = res

    res.sum().backward()

    fc_src = dglGat.fc_src.weight.grad
    fc_dest = dglGat.fc_dst.weight.grad

    return forward_correct, fc_src, fc_dest


class ToyModel(nn.Module):

    def __init__(self, gpu_id):
        super().__init__()
        self.ll = DistGATConv(4, 8, gpu_id,  num_heads=1)
        self.gpu_id = gpu_id

    def forward(self, bipartite_graph, f):
        return self.ll(bipartite_graph, f, self.gpu_id)


def test_dist_bipartite_process(proc_id, n_gpus):
    print("starting sub process", proc_id)
    dev_id = proc_id
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=proc_id)
    torch.cuda.set_device(dev_id)

    model = ToyModel(proc_id)

    model.ll.attn_l = torch.nn.Parameter(
        torch.ones(model.ll.attn_l.shape))
    print('x', model.ll.attn_l.requires_grad)
    model.ll.attn_r = torch.nn.Parameter(
        torch.ones(model.ll.attn_r.shape))
    model.ll.fc.weight = torch.nn.Parameter(
        torch.ones(model.ll.fc.weight.shape))

    model = model.to(dev_id)
    model = DistributedDataParallel(
        model, device_ids=[dev_id], output_device=dev_id)
    bg = get_bipartite_graph(proc_id)
    bg.to_gpu()
    f = torch.ones((2, 4), device=proc_id)

    out = model(bg, f)
    out.sum().backward()


def test_dist_bipartite():
    mp.set_start_method('spawn')
    print("Launch multiple gpus")
    n_gpus = 4
    procs = []
    for proc_id in range(4):
        p = mp.Process(target=(test_dist_bipartite_process),
                       args=(proc_id, n_gpus))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    test_base()
