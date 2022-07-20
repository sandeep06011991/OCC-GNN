import dgl
import torch.nn as nn
import torch
from torch.nn.parallel import gather
import time

class DistSageConv(nn.Module):

    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, aggregator_type = "gcn", \
        feat_drop=0.1, bias=True):
        super( DistSageConv, self).__init__()
        self.device_ids = [0,1,2,3]
        self._in_src_feats = in_feats
        # self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        # aggregator type: mean/pool/lstm/gcn
        assert aggregator_type in ['sum']
        self.fc = nn.Linear(self._in_src_feats * 2, out_feats)
        # repl_linear = torch.nn.parallel.replicate(self.fc.to(0),self.device_ids)
        # print("test replication")
        # if aggregator_type == 'pool':
        #     self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        # if aggregator_type != 'gcn':
        #     self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        # self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        # if bias:
        #     self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        # else:
            # self.register_buffer('bias', None)
        self.reset_parameters()
        #self.repl_linear = torch.nn.parallel.replicate(self.fc.to(0),self.device_ids)


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # if self._aggre_type == 'pool':
        #             nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        # if self._aggre_type != 'gcn':
        #     nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # torch.nn.init.ones_(self.fc.weight)

    def forward(self, bipartite_graphs, x):
        # distributed_graphs = [bipartite_graphs(4)]
        # distributed_tensor = [tensor(4)]
        # Replicate all linear modules
        # print("Starting first layer forward pass !!!! ")
        # x = [self.feat_drop(xx)  for xx in x ]
        # Compute H^{l+1}_n(i)
        # Assume aggregator is sum.
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start_shuffle = torch.cuda.Event(enable_timing=True)
        # end_shuffle = torch.cuda.Event(enable_timing=True)
        # start1 = torch.cuda.Event(enable_timing=True)
        # end1 = torch.cuda.Event(enable_timing=True)
        # start2 = torch.cuda.Event(enable_timing = True)
        # end2 = torch.cuda.Event(enable_timing = True)
        ng_gather = []
        # torch.cuda.set_device(0)
        # start1.record()
        # torch.cuda.nvtx.range_push("dist_aggr")
        # t0 = time.time()
        # torch.cuda.nvtx.range_push("gather_local")
        for src_gpu in range(4):
            ng_gather.append(bipartite_graphs[src_gpu].gather(x[src_gpu]))
        # torch.cuda.set_device(0)
        # end1.record()
        # torch.cuda.nvtx.range_pop()
        # t1 = time.time()
        # torch.cuda.nvtx.range_push("shuffle")
        shuffle_move_time = 0
        # torch.cuda.set_device(0)
        # start_shuffle.record()
        '''
        for src_gpu in range(4):
            for dest_gpu in range(4):
                t = bipartite_graphs[src_gpu].pull_for_remotes(ng_gather[src_gpu], dest_gpu)
                bipartite_graphs[dest_gpu].push_from_remotes(ng_gather[dest_gpu], t, src_gpu)
        '''
        # torch.cuda.set_device(0)
        # end_shuffle.record()
        # torch.cuda.nvtx.range_pop()
        # t2 = time.time()
        # print("shuffle", shuffle_move_time, "move time", t2 - t1)
        # concat(h^l_i,h^{l+1}_N(i))
        # torch.cuda.set_device(0)
        # start2.record()
        out1 = []
        for src_gpu in range(4):
            out1.append(torch.cat([bipartite_graphs[src_gpu].self_gather(x[src_gpu]), \
                        ng_gather[src_gpu]],dim = 1))
        out2 = []
        for src_gpu in range(4):
            out2.append(bipartite_graphs[src_gpu].\
                slice_owned_nodes(out1[src_gpu]))
        # print(self.fc.weight.shape,"Linear layear shape")
        # repl_linear = []
        # for i in range(4):
        #     print(i)
        #     repl_linear.append(self.fc.to(i))
        #     print(repl_linear[i].weight.device)
        # for i in range(4):
        #     torch.cuda.set_device(i)
        #     torch.cuda.synchronize()
        #     print(i,"device", repl_linear[i].weight.device)
        repl_linear = torch.nn.parallel.replicate(self.fc.to(0),self.device_ids)
        out3 = []
        for i in range(4):
            # print(i)
            # print(out2[i].shape)
            # print("apply lin layer",repl_linear[i].weight.device, out2[i].device)
            # out3.append(self.fc.to(i)(out2[i]))
            out3.append(repl_linear[i](out2[i]))
        t4 = time.time()
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.set_device(0)
        # end2.record()
        # torch.cuda.synchronize(end1)
        # torch.cuda.synchronize(end_shuffle)
        # torch.cuda.synchronize(end2)
        # print("first aggr layer",start1.elapsed_time(end1)/1000)
        # print("shuffle time",start_shuffle.elapsed_time(end_shuffle)/(1000))
        # print("last aggr layer", start2.elapsed_time(end2)/1000)
        return out3

    def faster_forward():
        pass
