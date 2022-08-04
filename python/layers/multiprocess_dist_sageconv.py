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
        feat_drop=0.1, bias=True, queues = None, deterministic = False):
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
        self.deterministic = deterministic
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
        if self.deterministic:
            self.fc1.weight = torch.nn.Parameter(
                torch.ones(self.fc1.weight.shape))
            self.fc2.weight = torch.nn.Parameter(
                torch.ones(self.fc2.weight.shape))
        else:
            nn.init.xavier_uniform_(self.fc2.weight,gain = gain)
            nn.init.xavier_uniform_(self.fc1.weight,gain = gain)


    def print_grad(self,l):
        if self.gpu_id != 0:
            return
        print("layer neigh ",l,self.fc1.weight[:3,0], torch.sum(self.fc1.weight))
        print("layer self",l,self.fc2.weight[:3,0], torch.sum(self.fc2.weight))
        if self.fc1.weight.grad != None:
            print("layer neigh grad",l,self.fc1.weight.grad[:3,0], torch.sum(self.fc1.weight.grad))
            print("layer self grad",l,self.fc2.weight.grad[:3,0], torch.sum(self.fc2.weight.grad))
        
    def old_forward(self, bipartite_graph,x):
        print(torch.sum(self.fc1.weight))
        out1 = bipartite_graph.gather(x)
        out2 = out1
        # out2 = Shuffle.apply(out1, self.queues, self.gpu_id, bipartite_graph.from_ids, bipartite_graph.to_ids)
        out3 = (torch.cat([bipartite_graph.self_gather(x), \
                        out2],dim = 1))
        out4 = (bipartite_graph.slice_owned_nodes(out3))

        if self._in_src_feats <= self._out_feats:
            out5 = self.fc(out4)

    def forward(self, bipartite_graph, x, l):
        t1 = time.time()
        # assert(torch.all(self.fc1.weight!=0))
        # assert(torch.all(self.fc2.weight!=0))
        # if self._in_src_feats  > self._out_feats:
        if False:
            out = self.fc1(x)
            out1 = bipartite_graph.gather(out)
            out2 = Shuffle.apply(out1, self.queues, self.gpu_id,bipartite_graph.to_ids, bipartite_graph.from_ids, l)
        else:
            # Only for determinism
            # x = torch.ones(x.shape).to(self.gpu_id)
            out = bipartite_graph.gather(x)
            out2 = Shuffle.apply(out, self.queues, self.gpu_id, bipartite_graph.to_ids, bipartite_graph.from_ids,l)
            # out2 = x
            # out2 = self.fc1(out2)
        t2 = time.time()
        out6_b = bipartite_graph.slice_owned_nodes(out2)
        out6 = out6_b/bipartite_graph.in_degree
        print("layer neighbor",l,torch.sum(out6))
        out6 = self.fc1(out6)
        # For bug fixing
        # return out6_b
        out3 = bipartite_graph.self_gather(x)
        out4 = bipartite_graph.slice_owned_nodes(out3)
        out5 = self.fc2(out4)
        print("self layer sum",l,torch.sum(out3),torch.sum(out4))
        # final = out5

        final = out5 + out6
        # final = out5 + out6
        # if (not torch.all(final.sum(1) != 0)):
        #     print("problem node in layer", l)
        #     index_id = torch.where(final.sum(1)==0)[0]
        #     print("neighbour",bipartite_graph.in_degree[index_id])
        #     print(bipartite_graph.out_nodes[index_id ])
        #     print("input",torch.where(torch.sum(x,1) == 0))
        #     # print(out5[index_id],out6[index_id], out4[index_id])
        # assert(torch.all(final.sum(1) != 0))
        # # t3 = time.time()
        # if self.gpu_id == 0:
        #     print("layer ",l,"out",final[:3,0])
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
    print("fc1 grad",model_saved.ll.fc1.weight.grad)
    print("fc2 grad",model_saved.ll.fc2.weight.grad)

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
