import torch.nn as nn
import torch

from layers.opt_shuffle import Shuffle


class DistGATConv(nn.Module):

    # Not exactly matching SageConv as normalization and activation as removed.
    def __init__(self, in_feats, out_feats, gpu_id,
     deterministic = False, num_heads=3, negative_slope=0.2):
        super(DistGATConv, self).__init__()
        self.gpu_id = gpu_id
        self.num_heads = num_heads
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.fc = nn.Linear(in_feats, out_feats*num_heads, bias=False)

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.attn_l, gain=gain)
        nn.init.xavier_uniform_(self.attn_r, gain=gain)

    def forward(self, bipartite_graphs, in_feats, l, in_degrees):
        src_prefix_shape = in_feats.shape[:-1]
        in_feats = self.fc(in_feats).view(
            *src_prefix_shape, self.num_heads, self.out_feats)
        el = (in_feats * self.attn_l).sum(dim=-1).unsqueeze(-1)
        in_feats_temp = in_feats.view(
            *src_prefix_shape, self.num_heads*self.out_feats)
        out_feats = bipartite_graphs.self_gather(in_feats_temp)
        out_feats = out_feats.view(
            *out_feats.shape[:-1], self.num_heads, self.out_feats)

        er = (out_feats * self.attn_r).sum(dim=-1).unsqueeze(-1)
        er = Shuffle.apply(
            er, None, self.gpu_id,   bipartite_graphs.to_ids, bipartite_graphs.from_ids, None)

        e = bipartite_graphs.apply_edge(el, er)
        e = self.leaky_relu(e)
        # TODO: fix exponent overflow
        exponent = e
        exponent = torch.exp(e)
        # assert(torch.all(not torch.isnan(e)))
        sum_exponent = bipartite_graphs.apply_node(exponent)
        print("Shuffle attempt", sum_exponent.shape)
        sum_exponent = Shuffle.apply(
            sum_exponent, None, self.gpu_id,   bipartite_graphs.to_ids, bipartite_graphs.from_ids, None)
        shape = sum_exponent.shape
        print("Shuffle 1 OK!")
        sum_exponent = bipartite_graphs.set_remote_data_to_zero(sum_exponent)
        print("Shuffle attempt")
        sum_exponent = Shuffle.apply(
            sum_exponent, None, self.gpu_id,   bipartite_graphs.from_ids, bipartite_graphs.to_ids, None)
        print("Shuffle 2 Ok", sum_exponent.shape)
        sum_exponent = bipartite_graphs.copy_from_out_nodes(sum_exponent)
        print("suffle 2.5", sum_exponent.shape)
        attention = exponent / sum_exponent
        out = bipartite_graphs.attention_gather(attention, in_feats)
        print("Checl",attention.shape, in_feats.shape , out.shape)
        for i in range(4):
            print("Skip shuffle",out.shape, bipartite_graphs.to_ids[i].shape, bipartite_graphs.from_ids[i].shape, self.gpu_id)
        print("shuffle 3 attempt")
        out = Shuffle.apply(
            out, None,self.gpu_id,   bipartite_graphs.to_ids, bipartite_graphs.from_ids, None)
        print("Shuffle 3 Ok")
        out = bipartite_graphs.slice_owned_nodes(out)
        out = out.flatten(1)

        return out
