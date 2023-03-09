import dgl
import torch.nn as nn
import torch
import time
import nvtx
from layers.shuffle import Shuffle
import dgl.nn.pytorch.conv.sageconv as sgc
import torch.multiprocessing as mp
from data.test_bipartite import get_local_bipartite_graph
from torch.nn.parallel import DistributedDataParallel


class DistSageConv(nn.Module):

    def _backward_hook(self, module, grad_input, other):
        print("Internal Hook !!!!!!!!!!!!!")
        for i in grad_input:
            print("grad input", i.shape)
    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, gpu_id,  num_gpus,\
        feat_drop=0.1, bias=True, deterministic = False, skip_shuffle = False):
        super( DistSageConv, self).__init__()
        self.device_ids = [0,1,2,3]
        self._in_src_feats = in_feats
        # self._in_dst_feats = in_feats
        self._out_feats = out_feats
        aggregator_type = "sum"
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.gpu_id = gpu_id
        self.skip_shuffle = skip_shuffle

        # aggregator type: mean/pool/lstm/gcn
        assert aggregator_type in ['sum']
        # self.fc = nn.Linear(self._in_src_feats * 2, out_feats)
        self.fc1 = nn.Linear(self._in_src_feats, out_feats, bias = False)
        self.fc2 = nn.Linear(self._in_src_feats, out_feats, bias = False)
        # self.deterministic = deterministic
        
        self.deterministic = deterministic
        self.reset_parameters()
        self.shuffle_time = 0
        self.num_gpus = num_gpus
        self.debug = False
        # self.local_stream = torch.cuda.default_stream()
        # self.remote_stream = torch.cuda.default_stream()
        # Not parallelizing in htis variation

        # self.local_stream = torch.cuda.Stream()
        # self.remote_stream = torch.cuda.Stream()

        # self.e1 = torch.cuda.Event(enable_timing = True)
        # self.e2 = torch.cuda.Event(enable_timing = True)
        # self.e3 = torch.cuda.Event(enable_timing = True)
        # self.e4 = torch.cuda.Event(enable_timing = True)
        # self.e5 = torch.cuda.Event(enable_timing = True)

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

    def get_reset_shuffle_time(self):
        ret = self.shuffle_time
        self.shuffle_time = 0
        return ret

    # Infeatures are post pulled.

    def forward(self, bipartite_graph, in_features, layer_id):
        t1 = time.time()
        if self.fc1.in_features > self.fc1.out_features:
            # Could incur more communication potentially
            # Makes backward pass mode complaceted
            out = self.fc1(in_features)
        else:
            out = in_features
            # t11 = time.time()
        # Note to self. These can be easily overlapped
        # However launching gather local and remote on different streams is easy in fp
        # Will break in the backawrd pass

        # self.local_stream.wait_stream(torch.cuda.current_stream())
        # self.remote_stream.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(self.remote_stream):
        torch.cuda.nvtx.range_push("gather_remote {}".format(self.gpu_id))
        out1 = bipartite_graph.gather_remote(out)
        torch.cuda.nvtx.range_pop()
        t1 = time.time()
        
        merge_tensors = Shuffle.apply(out1, self.gpu_id,self.num_gpus,  layer_id, bipartite_graph.get_from_nds_size(), \
                            bipartite_graph.to_offsets)
                    # Work on this signature later.
        # with torch.cuda.stream(self.local_stream):
        t2 = time.time()
        self.shuffle_time += (t2-t1)
        torch.cuda.nvtx.range_push("Gater local{}".format(self.gpu_id))
        out3 = bipartite_graph.gather_local(out).clone()
        torch.cuda.nvtx.range_pop()

        # torch.cuda.current_stream().wait_stream(self.local_stream)
        # torch.cuda.current_stream().wait_stream(self.remote_stream)
        # good practice, ensures caching allocator safety of memory created
        # on one stream and used on another
        # out.record_stream(self.local_stream)
        # out.record_stream(self.remote_stream)
        # out3.record_stream(torch.cuda.current_stream())
        torch.cuda.nvtx.range_push("Range merge")
        if not self.skip_shuffle:
            for i in range(self.num_gpus):
                if i != self.gpu_id:
                    out3[bipartite_graph.from_ids[i]] += merge_tensors[i]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Clear crap")
        if self.debug:
            assert(not torch.any(torch.isnan(out3)))
            assert(torch.all(bipartite_graph.out_degrees != 0))
        torch.cuda.nvtx.range_pop()
        a = bipartite_graph.out_degrees.shape
        degree = bipartite_graph.out_degrees.reshape(a[0],1)
        out4 = (out3/degree)
        if self.debug:
            assert(not torch.any(torch.isnan(out4)))

        if not self.fc1.in_features > self.fc1.out_features:
            out4 = self.fc1(out4)
        # t22 = time.time()
        out5 = bipartite_graph.self_gather(in_features)
        out6 = self.fc2(out5)
        final = out4 + out6
        # print("reamining", c-b)
        return final


def test_base():
    src_ids = []
    dest_ids = []
    for dest in range(4):
        for source in range(8):
            src_ids.append(source)
            dest_ids.append(dest)

    g = dgl.create_block((src_ids, dest_ids), 8, 4)
    dglSage = dgl.nn.SAGEConv(8,4, 'mean')
    dglSage.fc_self.weight = torch.nn.Parameter(
        torch.ones(dglSage.fc_self.weight.shape))
    dglSage.fc_neigh.weight = torch.nn.Parameter(
        torch.ones(dglSage.fc_neigh.weight.shape))

    ones = f = torch.ones((8,8), requires_grad = True)
    res = dglSage(g, ones)
    forward_correct = res
    res.sum().backward()
    fc1_grad = dglSage.fc_self.weight.grad
    fc2_grad = dglSage.fc_neigh.weight.grad
    print("FP Result", forward_correct)
    print("BP Grad", ones.grad)
    # print(fc1_grad,fc2_grad,forward_correct)
    return forward_correct, fc1_grad, fc2_grad
#
class ToyModel(nn.Module):

    def __init__(self,gpu_id):
        super().__init__()
        self.ll = DistSageConv(8000,8000,gpu_id,deterministic = True)
        self.ll.reset_parameters()

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
    model = model.to(dev_id)
    model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    bg = get_local_bipartite_graph(proc_id)
    f = torch.ones((2,8000),device = proc_id, requires_grad = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    for i in range(1):
        e1.record()
        out = model.forward(bg,f)
        out.sum().backward()
        e2.record()
        e2.synchronize()
        print("time",e1.elapsed_time(e2)/1000)
    print(f.grad, out)

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
    test_dist_bipartite()
    # test_base()
