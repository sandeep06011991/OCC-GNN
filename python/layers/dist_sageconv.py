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
        self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        # aggregator type: mean/pool/lstm/gcn
        assert aggregator_type in ['mean','pool','gcn']
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
                    nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, bipartite_graph, shuffle_matrix, x):
        # distributed_graphs = [bipartite_graphs(4)]
        # distributed_tensor = [tensor(4)]
        # Replicate all linear modules
        print("Starting first layer forward pass !!!! ")
        x = [self.feat_drop(xx)  for xx in x ]
        # Compute H^{l+1}_n(i)
        # Assume aggregator is sum.
        out = []
        for src_gpu in bipartite_graphs.keys():
            out.append(bipartite_graphs[src_gpu].gather(x[gather]))
        for src_gpu in shuffle_matrix.keys():
            for dest_gpu in shuffle_matrix[src_gpu].keys():
                t = bipartite_graph[src_gpu].pull_from_remotes(out[src_gpu], \
                    shuffle_matrix[src_gpu][dest_gpu])
                bipartite_graph[dest_gpu].push_from_remotes(out[dest_gpu],t)
        # concat(h^l_i,h^{l+1}_N(i))
        for src_gpu in shuffle_matrix.keys():
            out.append(torch.concat(bipartite_graphs[src_gpu].gather(x[src_gpu]), \
                        bipartite_graphs[src_gpu].gather(x[src_gpu]))))
        #
        repl_linear = self.replicate(self.fc,self.devices_ids)
        out = []
        for i in range(4):
            out.append(repl_linear[i](out[self.fc]))

    def apply_message_passing(self, distributed_graphs, distributed_input):
        out = []
        # Gather
        for i in range(4):
            graph = distributed_graphs[i]
            out.append(graph.gather(distributed_input[i]))
            return out
        # Shuffle Indices
        for src_id in range (4):
            for dest_id, src, dest in graph.shuffle_pairs:
                out[dest_id][dest] += distributed_input[src_id][offsets].to(dest_id)
        return out

    def apply_replica_on_tensor(self, replica_nn, dist_tensors):
        out = []
        for i in range(4):
            out.append(replica_nn[i](dist_tensors[i]))
        return out
