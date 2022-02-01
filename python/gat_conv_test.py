import dgl
import torch
import torch.nn as nn
# from dgl.nn import GATConv

# src_ids = torch.tensor([2, 3, 4])
# # Destination nodes for edges (2, 1), (3, 2), (4, 3)
# dst_ids = torch.tensor([1, 2, 3])
# g = dgl.graph((src_ids, dst_ids))
# g.ndata["features"] = torch.rand(5,10)
# # gatconv(graph,input_features)
#
# layer = GATConv(10, 2, num_heads=3)
# g = g.add_self_loop()
#
# class GraphWrapper():
#     pass
#
# class DistTensorWrapper():
#     pass

# m = torch.nn.Linear(10,10)
class DistLinear(nn.Module):
    def __init__(self,inf,outs):
        l = torch.nn.Linear(inf,outs)
        self.layers = [l.clone().to(i) for i in range(4)]

    def forward(self,ls):
        pass
# ms = [m.to(i) for i in range(4)]
ds = DistLinear(10,10)

# def forward(self, graph, feat, get_attention=False):
#         r"""
#         Description
#         -----------
#         Compute graph attention network layer.
#         Parameters
#         ----------
#         graph : DGLGraph
#             The graph.
#         feat : torch.Tensor or pair of torch.Tensor
#             If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
#             :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
#             If a pair of torch.Tensor is given, the pair must contain two tensors of shape
#             :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
#         get_attention : bool, optional
#             Whether to return the attention values. Default to False.
#
#         Returns
#         -------
#         torch.Tensor
#             The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
#             is the number of heads, and :math:`D_{out}` is size of output feature.
#         torch.Tensor, optional
#             The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
#             edges. This is returned only when :attr:`get_attention` is ``True``.
#
#         Raises
#         ------
#         DGLError
#             If there are 0-in-degree nodes in the input graph, it will raise DGLError
#             since no message will be passed to those nodes. This will cause invalid output.
#             The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
#         """
#         with graph.local_scope():
#             if not self._allow_zero_in_degree:
#                 if (graph.in_degrees() == 0).any():
#                     raise DGLError('There are 0-in-degree nodes in the graph, '
#                                    'output for those nodes will be invalid. '
#                                    'This is harmful for some applications, '
#                                    'causing silent performance regression. '
#                                    'Adding self-loop on the input graph by '
#                                    'calling `g = dgl.add_self_loop(g)` will resolve '
#                                    'the issue. Setting ``allow_zero_in_degree`` '
#                                    'to be `True` when constructing this module will '
#                                    'suppress the check and let the code run.')
#
#                 h_src = h_dst = self.feat_drop(feat)
#                 feat_src = feat_dst = self.fc(h_src).view(
#                     -1, self._num_heads, self._out_feats)
#                 if graph.is_block:
#                     feat_dst = feat_src[:graph.number_of_dst_nodes()]
#             # NOTE: GAT paper uses "first concatenation then linear projection"
#             # to compute attention scores, while ours is "first projection then
#             # addition", the two approaches are mathematically equivalent:
#             # We decompose the weight vector a mentioned in the paper into
#             # [a_l || a_r], then
#             # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
#             # Our implementation is much efficient because we do not need to
#             # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
#             # addition could be optimized with DGL's built-in function u_add_v,
#             # which further speeds up computation and saves memory footprint.
#             el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
#             er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
#             graph.srcdata.update({'ft': feat_src, 'el': el})
#             graph.dstdata.update({'er': er})
#             # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
#             graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
#             e = self.leaky_relu(graph.edata.pop('e'))
#             # compute softmax
#             graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
#             # message passing
#             graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
#                              fn.sum('m', 'ft'))
#             rst = graph.dstdata['ft']
#             # residual
#             if self.res_fc is not None:
#                 resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
#                 rst = rst + resval
#             # bias
#             if self.bias is not None:
#                 rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
#             # activation
#             if self.activation:
#                 rst = self.activation(rst)
#
#             if get_attention:
#                 return rst, graph.edata['a']
#             else:
#                 return rst
# Test 1:
# out = layer(g,g.ndata["features"])
# print(out)

    # get forward pass running naive
# Create distributed create_heterograph
# Create distributed gat conv

# Fix forward pass
# Fix backward pass
