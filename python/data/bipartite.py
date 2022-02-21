import torch
import dgl
import dgl.function as fn

class BipartiteGraph:

    def __init__(self, edge_src_u, edge_dest_v):
        # To enforce consistency across layers
        # U and V have to be in the same order
        reordered_to_orig_src,reordered_src = torch.unique(edge_src_u,return_inverse = True)
        reordered_to_orig_dest,reordered_dest = torch.unique(edge_dest_v,return_inverse = True)
        # Handle self edges.
        self.graph = dgl.heterograph({('_U','_E','_V'):(reordered_src,reordered_dest)})
        # assert(self.graph.in_degree(0) == 0)
        self.num_nodes_u = self.graph.num_nodes('_U')
        self.num_nodes_v = self.graph.num_nodes('_V')
        assert(self.num_nodes_u > self.num_nodes_v)

    def gather(self, f_in):
        with self.graph.local_scope():
            assert(f_in.shape[0] == self.graph.number_of_nodes('_U'))
            self.graph.nodes['_U'].data['in'] = f_in
            self.graph.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            return self.graph.nodes['_V'].data['out']

    def attention_gather(self, inf):
        with self.graph.local_scope():
            self.graph.nodes['_V'].data['in_v'] = inf[:self.num_nodes_v]
            self.graph.nodes['_U'].data['in'] = inf
            self.graph.update_all(fn.u_mul_v('in','in_v','h'),fn.sum('h','out'))
            return self.graph.nodes['_V'].data['out']

    def self_gather(self, f_in):
        raise Exception("Not implemented")


if __name__ == "__main__":
    g = BipartiteGraph([0, 0, 0, 1, 2, 2, 3], [0, 1, 1, 1, 2, 2, 2])
    inf = torch.rand(g.num_nodes_u,requires_grad = True)
    out = g.gather(inf)
    out.sum().backward()
    print("Test all gradient flows of bipartite graphs.")
    print("This can be a simple wrapper.")
