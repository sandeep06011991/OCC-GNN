from dgl.nn import expand_as_pair
import torch.nn as nn
import torch
from torch.nn.parallel import gather
import time

from python.layers.opt_shuffle import Shuffle


class DistSageConv(nn.Module):

    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, num_heads=1):
        super(DistSageConv, self).__init__()

        self.num_heads = num_heads
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.in_feats, gain=gain)
        nn.init.xavier_uniform_(self.out_feats, gain=gain)

    def forward(self, bipartite_graphs, x):
        el = (self.in_feats * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (self.out_feats * self.attn_r).sum(dim=-1).unsqueeze(-1)
        er = Shuffle(er, bipartite_graphs.from_ids, bipartite_graphs.to_ids)

        e = self.bipartite_graphs.apply_edge(el, er)

        e = self.leaky_relu(e)
        exponent = torch.exp(e)
        sum_exponent = self.bipartite_graphs.apply_node(exponent)
        sum_exponent = Shuffle(
            sum_exponent, bipartite_graphs.from_ids, bipartite_graphs.to_ids)
        sum_exponent = bipartite_graphs.copy_from_out_nodes(sum_exponent)

        attention = exponent / sum_exponent

        out = bipartite_graphs.attention_gather(attention, x)
        return out
