import dgl
import torch.nn as nn
import torch
from torch.nn.parallel import gather

class DistSageConv(nn.Module):

    def __init__(self, in_feats, out_feats, aggregator_type = "gcn", \
        feat_drop=0., bias=True, norm=None, activation=None):
        super( DistSageConv, self).__init__()
        self.device_ids = [0,1,2,3]
        self._in_src_feats = in_feats
        self._in_dst_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
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

    def forward(self, distributed_graphs, distributed_input):
        # distributed_graphs = [bipartite_graphs(4)]
        # distributed_tensor = [tensor(4)]
        # Replicate all linear modules
        replica_feat_drop = self.replicate(self.feat_drop, self.device_ids)
        if aggregator_type == 'pool':
            replica_fc_pool = self.replicate(self.fc_pool,self.device_ids)
        if aggregator_type != 'gcn':
            replica_fc_self = self.replicate(self.fc_self, self.device_ids)
        replica_fc_neigh = self.replicate(self.fc_neigh, self.device_ids)
        if bias:
            replica_bias = self.replicate(self.bias, self.device_ids)
        feat_src = feat_dst = self.apply_replica_on_tensor(replica_feat_drop, distributed_input)
        # msg_fn = fn.copy_src('h','m')


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


if __name__ == "__main__":
    g = DistSageConv(10,10)
