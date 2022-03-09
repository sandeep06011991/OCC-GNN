import dgl
import torch.nn as nn
import torch
from torch.nn.parallel import gather

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

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # if self._aggre_type == 'pool':
        #             nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        # if self._aggre_type != 'gcn':
        #     nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        # torch.nn.init.ones_(self.fc.weight)

    def forward(self, bipartite_graphs, shuffle_matrix,owned_nodes,  x):
        # distributed_graphs = [bipartite_graphs(4)]
        # distributed_tensor = [tensor(4)]
        # Replicate all linear modules
        # print("Starting first layer forward pass !!!! ")
        # x = [self.feat_drop(xx)  for xx in x ]
        # Compute H^{l+1}_n(i)
        # Assume aggregator is sum.
        ng_gather = []
        for src_gpu in bipartite_graphs.keys():
            ng_gather.append(bipartite_graphs[src_gpu].gather(x[src_gpu]))

        for src_gpu in shuffle_matrix.keys():
            for dest_gpu in shuffle_matrix[src_gpu].keys():
                t = bipartite_graphs[src_gpu].pull_for_remotes(ng_gather[src_gpu], \
                    shuffle_matrix[src_gpu][dest_gpu].to(src_gpu))
                bipartite_graphs[dest_gpu].push_from_remotes(ng_gather[dest_gpu],t.to(dest_gpu), \
                    shuffle_matrix[src_gpu][dest_gpu].to(dest_gpu) )
        # concat(h^l_i,h^{l+1}_N(i))
        out1 = []
        for src_gpu in bipartite_graphs.keys():
            out1.append(torch.cat([bipartite_graphs[src_gpu].self_gather(x[src_gpu]), \
                        ng_gather[src_gpu]],dim = 1))
        out2 = []
        for src_gpu in bipartite_graphs.keys():
            out2.append(bipartite_graphs[src_gpu].\
                slice_owned_nodes(out1[src_gpu],owned_nodes[src_gpu].to(src_gpu)))
        repl_linear = torch.nn.parallel.replicate(self.fc.to(0),self.device_ids)
        out3 = []
        for i in range(4):
            out3.append(repl_linear[i](out2[i]))
        return out3
