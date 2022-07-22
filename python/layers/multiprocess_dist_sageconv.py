import dgl
import torch.nn as nn
import torch
from torch.nn.parallel import gather
import time
from layers.opt_shuffle import Shuffle
import dgl.nn.pytorch.conv.sageconv as sgc
import torch.multiprocessing as mp
from data.test_bipartite import get_bipartite_graph
from torch.nn.parallel import DistributedDataParallel

class DistSageConv(nn.Module):

    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, gpu_id, aggregator_type = "gcn", \
        feat_drop=0.1, bias=True, queues = None):
        super( DistSageConv, self).__init__()
        self.device_ids = [0,1,2,3]
        self._in_src_feats = in_feats
        # self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.queues = queues
        self.gpu_id = gpu_id
        # aggregator type: mean/pool/lstm/gcn
        assert aggregator_type in ['sum']
        # self.fc = nn.Linear(self._in_src_feats * 2, out_feats)
        self.fc1 = nn.Linear(self._in_src_feats, out_feats,bias=False)

        self.fc2 = nn.Linear(self._in_src_feats, out_feats,bias=False)
        self.reset_parameters()
        # self.sgc = sgc.SAGEConv(self._in_src_feats, out_feats, aggregator_type = 'mean')
        # if aggregator_type == 'pool':
        #     self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        # if aggregator_type != 'gcn':
        #     self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        # self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        # if bias:
        #     self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        # else:
            # self.register_buffer('bias', None)



    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=gain)

    def old_forward(self, bipartite_graph,x):
        out1 = bipartite_graph.gather(x)
        out2 = out1
        # out2 = Shuffle.apply(out1, self.queues, self.gpu_id, bipartite_graph.from_ids, bipartite_graph.to_ids)
        out3 = (torch.cat([bipartite_graph.self_gather(x), \
                        out2],dim = 1))
        out4 = (bipartite_graph.slice_owned_nodes(out3))

        if self._in_src_feats <= self._out_feats:
            out5 = self.fc(out4)

    def forward(self, bipartite_graph, x, l):
        # y =torch.rand((num_nodes,x.shape[1]),device = x.device)
        # t0 = time.time()
        # num_nodes = bipartite_graph.num_nodes_v
        # out = self.sgc(bipartite_graph.graph,(x,y)))
        t1 = time.time()
        # print(t1-t0,"target time")
        # out = bipartite_graph.slice_owned_nodes(out)
        # return out
        print(torch.sum(self.fc1.weight))
        if self._in_src_feats  > self._out_feats:
            out = self.fc1(x)
            out1 = bipartite_graph.gather(out)
            # out2 = out1
            out2 = Shuffle.apply(out1, self.queues, self.gpu_id, bipartite_graph.from_ids, bipartite_graph.to_ids,l)
        else:
            out = bipartite_graph.gather(x)
            out2 = Shuffle.apply(out, self.queues, self.gpu_id, bipartite_graph.from_ids, bipartite_graph.to_ids,l)
            # out2 = out
            out2 = self.fc1(out2)
        t2 = time.time()
        out3 = bipartite_graph.self_gather(x)
        out4 = bipartite_graph.slice_owned_nodes(out3)
        out5 = self.fc2(out4)
        out6 = bipartite_graph.slice_owned_nodes(out2)
        final = out5 + out6
        t3 = time.time()
        # if out.device == torch.device(2):
        #     print(t3-t2,"second half of nn",l,"layer",out.device,"device")
        #     print(t2-t1,"first half of nn",l,"layer",out.device,"device")
        return final

def test_base():
    src_ids = []
    dest_ids = []
    for dest in range(4):
        for source in range(8):
            src_ids.append(source)
            dest_ids.append(dest)

    g = dgl.create_block((src_ids, dest_ids), 8, 4)
    dglSage = dgl.nn.SAGEConv(4, 8, 'mean')
    dglSage.fc_self.weight = torch.nn.Parameter(
        torch.ones(dglSage.fc_self.weight.shape))
    dglSage.fc_neigh.weight = torch.nn.Parameter(
        torch.ones(dglSage.fc_neigh.weight.shape))

    ones = torch.ones(8)
    res = dglSage(g, ones)
    forward_correct = res
    res.sum().backward()
    fc1_grad = dglSage.fc_self.weight.grad
    fc2_grad = dglSage.fc_neigh.weight.grad
    print(fc1_grad,fc2_grad,forward_correct)
    return forward_correct, fc1_grad, fc2_grad

class ToyModel(nn.Module):

    def __init__(self,gpu_id):
        super().__init__()
        self.ll = DistSageConv(8,4,gpu_id,aggregator_type = "sum")

    def forward(self,bipartite_graph,f):
        return self.ll(bipartite_graph,f,0)

def test_dist_bipartite_process(proc_id,n_gpus):
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
    model.ll.fc1.weight = torch.nn.Parameter(torch.ones(model.ll.fc1.weight.shape))
    model.ll.fc2.weight = torch.nn.Parameter(torch.ones(model.ll.fc1.weight.shape))
    model_saved = model
    model = model.to(dev_id)
    model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    bg = get_bipartite_graph(proc_id)
    bg.to_gpu()
    f = torch.ones((2,8),device = proc_id)
    out = model(bg,f)

    print(out)
    out.sum().backward()
    print(model_saved.ll.fc1.weight.grad)
    print(model_saved.ll.fc2.weight.grad)

def test_dist_bipartite():
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
    # test_dist_bipartite()
    test_base()
