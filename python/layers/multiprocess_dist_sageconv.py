import dgl
import torch.nn as nn
import torch
from torch.nn.parallel import gather
import time
from layers.opt_shuffle import Shuffle
import dgl.nn.pytorch.conv.sageconv as sgc
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
        self.fc1 = nn.Linear(self._in_src_feats, out_feats)
        self.fc2 = nn.Linear(self._in_src_feats, out_feats)
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

def get_base():
    src_ids = []
    dest_ids = []
    for dest in range(4):
        for source in range(8):
            src_ids.append(source)
            dest_ids.append(dest)

    g = dgl.create_block((src_ids, dest_ids), 8, 4)
    dglSage = SAGEConv(4, 8, 'mean')
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
    return forward_correct, fc1_grad, fc2_grad

if __name__ == "__main__":
    unit_test()
