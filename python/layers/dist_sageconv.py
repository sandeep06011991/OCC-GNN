import dgl
import torch.nn as nn
import torch
import time
from layers.opt_shuffle import Shuffle
import dgl.nn.pytorch.conv.sageconv as sgc
import torch.multiprocessing as mp
from data.test_bipartite import get_dummy_bipartite_graph
from torch.nn.parallel import DistributedDataParallel

class DistSageConv(nn.Module):

    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, gpu_id, \
        feat_drop=0.1, bias=True, deterministic = False):
        super( DistSageConv, self).__init__()
        self.device_ids = [0,1,2,3]
        self._in_src_feats = in_feats
        # self._in_dst_feats = in_feats
        self._out_feats = out_feats
        aggregator_type = "sum"
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.gpu_id = gpu_id

        # aggregator type: mean/pool/lstm/gcn
        assert aggregator_type in ['sum']
        # self.fc = nn.Linear(self._in_src_feats * 2, out_feats)
        self.fc1 = nn.Linear(self._in_src_feats, out_feats, bias = False)
        self.fc2 = nn.Linear(self._in_src_feats, out_feats, bias = False)
        # self.deterministic = deterministic
        self.deterministic = deterministic
        self.reset_parameters()
        self.local_stream = torch.cuda.Stream()
        self.remote_stream = torch.cuda.Stream()

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


    # Infeatures are post pulled.
    def forward(self, bipartite_graph, in_features, layer_id):
        t1 = time.time()
        if self.fc1.in_features > self.fc1.out_features:
            # Could incur more communication potentially
            # Makes backward pass mode complaceted
            # t00 = time.time()
            out = self.fc1(in_features)
        else:
            out = in_features
            # t11 = time.time()
        with torch.cuda.stream(self.remote_stream):
                out1 = bipartite_graph.gather_remote(out)
                # Work on this signature later.
                merge_tensors = Shuffle.apply(self.gpu_id, from_tensors = bipartite_graph.from_tensors(out1), \
                                    to_tensors = bipartite_graph.to_tensors())
        with torch.cuda.stream(self.local_stream):
            out3 = bipartite_gather.gather_local(out)
        self.local_stream.synchronize()
        self.remote_stream.synchronize()
        out4 = bipartite_graph.shuffle_merge(out3, merge_tensors)

        assert(not torch.any(torch.isnan(out4)))
        assert(torch.all(bipartite_graph.in_degree != 0))
        a = bipartite_graph.in_degree.shape
        degree = bipartite_graph.in_degree.reshape(a[0],1)
        out6 = (out6_b/degree)
        assert(not torch.any(torch.isnan(out6)))

        if not self.fc1.in_features > self.fc1.out_features:
            out6 = self.fc1(out6)
        # t22 = time.time()
        out3 = bipartite_graph.self_gather(in_feats)
        out4 = self.fc2(out3)
        final = out4 + out6
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
