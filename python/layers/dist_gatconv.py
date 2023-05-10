import torch.nn as nn
import torch
import torch.multiprocessing as mp

from layers.shuffle import Shuffle
from layers.ref_shuffle import ShuffleRev

import dgl.nn.pytorch.conv.sageconv as sgc
import torch.multiprocessing as mp
from data.test_bipartite import get_local_bipartite_graph
from torch.nn.parallel import DistributedDataParallel

# from layers.opt_shuffle import Shuffle
# from layers.max_shuffle import ShuffleMax
import dgl
import time

class DistGATConv(nn.Module):

    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, gpu_id,  num_gpus,
     deterministic = False, skip_shuffle = False,  num_heads=3, negative_slope=0.2):
        super(DistGATConv, self).__init__()
        self.gpu_id = gpu_id
        self.num_heads = num_heads
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc = nn.Linear(in_feats, out_feats*num_heads, bias=False)
        self.skip_shuffle = skip_shuffle
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.deterministic =  deterministic
        self.reset_parameters()
        self.shuffle_time = 0
        self.num_gpus = num_gpus

    def get_reset_shuffle_time(self):
        ret = self.shuffle_time
        self.shuffle_time = 0
        return ret

    def reset_parameters(self):
        if self.deterministic:
            print("Forward ones")
            self.fc.weight = torch.nn.Parameter(
                torch.ones(self.fc.weight.shape))
            self.attn_l = torch.nn.Parameter(
                torch.ones(self.attn_l.shape))
            self.attn_r = torch.nn.Parameter(
                torch.ones(self.attn_l.shape))
        else:
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(self.attn_l, gain=gain)
            nn.init.xavier_uniform_(self.attn_r, gain=gain)
            nn.init.xavier_uniform_(self.fc.weight,gain = gain)

    def get_and_reset_shuffle_time(self):
        ret = self.shuffle_time
        self.shuffle_time = 0
        return ret

    def forward(self, bipartite_graph, in_feats, l, testing = False):
        # Refactor this increase readability.
        # Go for a cleanr convention. Feels very bad.
        # print(self.skip_shuffle)
        src_prefix_shape = in_feats.shape[:-1]
        in_feats = self.fc(in_feats).view(
            *src_prefix_shape, self.num_heads, self.out_feats)
        el = (in_feats * self.attn_l).sum(dim=-1).unsqueeze(-1)
        in_feats_temp = in_feats.view(
            *src_prefix_shape, self.num_heads*self.out_feats)
        out_feats = bipartite_graph.self_gather(in_feats_temp)
        out_feats = out_feats.view(
            *out_feats.shape[:-1], self.num_heads, self.out_feats)

        er = (out_feats * self.attn_r).sum(dim=-1).unsqueeze(-1)
        async_op = {}
        # Apply Edge here
        if not testing and not self.skip_shuffle:
            t1 = time.time()
            er_remote = ShuffleRev.apply(er, self.gpu_id,  self.num_gpus,  l, bipartite_graph.from_ids, \
                                bipartite_graph.to_offsets )
            t2 = time.time()
            self.shuffle_time += (t2-t1)

        e = bipartite_graph.apply_edge_local(el, er)
        e = self.leaky_relu(e)
        print("Till here is perfectly non blockign !!!!")
        if not testing and not self.skip_shuffle:
            e_r = bipartite_graph.apply_edge_remote(el,er_remote)
            e_r = self.leaky_relu(e_r)
        # TODO: fix exponent overflow
        # print("fix exponent overflow here.")
        #with torch.no_grad():
        if True:
            local_max = bipartite_graph.gather_local_max(e)
            if not testing and not self.skip_shuffle:
                remote_max = bipartite_graph.gather_remote_max(e_r)
                t1 = time.time()
                merge_maxes = Shuffle.apply(
                    remote_max, self.gpu_id, self.num_gpus,  l,  bipartite_graph.get_from_nds_size(),  bipartite_graph.to_offsets, None, None, async_op)
                t2 = time.time()
                self.shuffle_time += (t2-t1)
                for i in range(self.num_gpus):
                    if i != self.gpu_id:
                        local_max[bipartite_graph.from_ids[i]] = torch.max(local_max[bipartite_graph.from_ids[i]], merge_maxes[i])
            else:
                pass
                # global_max = local_max

            if not testing and not self.skip_shuffle:
                t1 = time.time()
                remote_max  = ShuffleRev.apply(
                    local_max , self.gpu_id, self.num_gpus, l,   bipartite_graph.from_ids,  bipartite_graph.to_offsets)
                t2 = time.time()
                self.shuffle_time += (t2-t1)
            local_max = bipartite_graph.copy_from_out_nodes_local(local_max)
            if not testing and not self.skip_shuffle:
                remote_max = bipartite_graph.copy_from_out_nodes_remote(remote_max)

        # m = 0

        exponent_l = e - local_max
        exponent_l = torch.exp(exponent_l)
        sum_exponent_local = bipartite_graph.apply_node_local(exponent_l)
        if not testing and not self.skip_shuffle:
            exponent_r = e_r - remote_max
            exponent_r = torch.exp(exponent_r)
            sum_exponent_remote_r = bipartite_graph.apply_node_remote(exponent_r)

        if not testing and not self.skip_shuffle:
            t1 = time.time()
            merge_sum = Shuffle.apply(
                sum_exponent_remote_r,self.gpu_id,  self.num_gpus, l,  bipartite_graph.get_from_nds_size(), bipartite_graph.to_offsets, None, None, async_op)
            t2 = time.time()
            self.shuffle_time += (t2-t1)
            sum_exponent_local = sum_exponent_local.clone()
            for i in range(self.num_gpus):
                if i != self.gpu_id:
                    sum_exponent_local[bipartite_graph.from_ids[i]] += merge_sum[i]
            t1 = time.time()
            remote_sum = ShuffleRev.apply(
                sum_exponent_local,  self.gpu_id, self.num_gpus,  l, bipartite_graph.from_ids,  bipartite_graph.to_offsets)
            t2 = time.time()
            self.shuffle_time += (t2-t1)
        sum_exponent = bipartite_graph.copy_from_out_nodes_local(sum_exponent_local)
        sum_exponent[torch.where(sum_exponent == 0)[0]] = 1
        attention_l = exponent_l / sum_exponent
        if not testing and not self.skip_shuffle:
            remote_sum_l = bipartite_graph.copy_from_out_nodes_remote(remote_sum)
            remote_sum_l[torch.where(remote_sum_l == 0)[0]] = 1
            attention_r = exponent_r / remote_sum_l

        out_local = bipartite_graph.attention_gather_local(attention_l, in_feats)
        if not testing and not self.skip_shuffle:
            out_remote = bipartite_graph.attention_gather_remote(attention_r, in_feats)
            t1 = time.time()
            merge_out = Shuffle.apply(
                out_remote, self.gpu_id, self.num_gpus,  l, bipartite_graph.get_from_nds_size(),bipartite_graph.to_offsets, None, None,async_op)
            t2 = time.time()
            self.shuffle_time += (t2-t1)

            out_local = out_local.clone()
            for i in range(self.num_gpus):
                if i != self.gpu_id:
                    out_local[bipartite_graph.from_ids[i]] += merge_out[i]

        out_local = out_local.flatten(1)
        return out_local




def test_base():
    src_ids = []
    dest_ids = []
    for dest in range(4):
        for source in range(8):
            src_ids.append(source)
            dest_ids.append(dest)

    g = dgl.create_block((src_ids, dest_ids), 8, 4)
    dglSage = dgl.nn.GATConv(8,4,  3)
    dglSage.fc.weight = torch.nn.Parameter(
        torch.ones(dglSage.fc.weight.shape))
    dglSage.attn_l = torch.nn.Parameter(torch.ones(dglSage.attn_l.shape))
    dglSage.attn_r = torch.nn.Parameter(torch.ones(dglSage.attn_r.shape))

    # dglSage.fc_neigh.weight = torch.nn.Parameter(
    #     torch.ones(dglSage.fc_neigh.weight.shape))

    ones = f = torch.ones((8,8), requires_grad = True)
    res = dglSage(g, ones)
    forward_correct = res
    res.sum().backward()
    # fc1_grad = dglSage.fc_self.weight.grad
    # fc2_grad = dglSage.fc_neigh.weight.grad
    print("FP Result", forward_correct)
    print("BP Grad", ones.grad)
    # print(fc1_grad,fc2_grad,forward_correct)
    # return forward_correct, fc1_grad, fc2_grad
#
class ToyModel(nn.Module):

    def __init__(self,gpu_id):
        super().__init__()
        self.ll = DistGATConv(8,4,gpu_id,deterministic = True)
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
    f = torch.ones((2,8),device = proc_id, requires_grad = True)
    for i in range(1):
        out = model(bg,f)
        print("Forward Pass",out)
        out.sum().backward()
        # model.module.ll.local_stream.synchronize()
        # model.module.ll.remote_stream.synchronize()
        #
        print("Backward Pass",f.grad)

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
