import torch
import dgl
import dgl.function as fn

'''
Features:
    Graphs are created using global ids.
    Inner layers (except last layer)
    reorder the ids to create a continuos block of feature output.
    Supported operations.
    1. Local Gather: returns sum of neigbours.
    2. Attention gather: returns sum of {F[u] * F[v] : v in N(u)}
    3. pull_from_self:
    4. push_from_remotes.
    This class stores information only local to the graph.
    All information regarding remotes must be handled elsewhere
    src should not be reordered in the last layer.
    Each gpu already contains a global to local order which will be used.
'''
class BipartiteGraph:

    def __init__(self, edge_src_u, edge_dest_v, device, global_to_local = None, local_to_global = None):
        # To enforce consistency across layers
        # U and V have to be in the same order
        # Assume self edges have no duplicates
        edge_src_u = edge_src_u.to(device)
        edge_dest_v = edge_dest_v.to(device)
        self_edges = torch.where(edge_src_u == edge_dest_v)[0]
        # assert(torch.unique(self_edges[edge_src_u]).shape == self_edges.shape)
        if global_to_local == None:
            reordered_to_orig_src,reordered_src = torch.unique(edge_src_u,return_inverse = True)
        else:
            reordered_to_orig_src = local_to_global
            reordered_src = global_to_local[edge_src_u]

        reordered_to_orig_dest,reordered_dest = torch.unique(edge_dest_v,return_inverse = True)
        # Handle self edges.
        self.self_u = reordered_src[self_edges]
        self.self_v = reordered_dest[self_edges]
        self.local_to_global_src = reordered_to_orig_src
        self.local_to_global_dest = reordered_to_orig_dest.to(device)
        self.graph = dgl.heterograph({('_U','_E','_V'):(reordered_src,reordered_dest)},device = device)
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

    def attention_gather(self, v_in, u_in):
        with self.graph.local_scope():
            self.graph.nodes['_V'].data['in_v'] = v_in
            self.graph.nodes['_U'].data['in'] = u_in
            self.graph.update_all(fn.u_mul_v('in','in_v','h'),fn.sum('h','out'))
            return self.graph.nodes['_V'].data['out']

    def self_gather(self, local_in):
        with self.graph.local_scope():
            out = torch.zeros(self.num_nodes_v,local_in.shape[1],device =  self.graph.device)
            out[self.self_v] = local_in[self.self_u]
            return out
    def pull_from_remotes(self,local_out, global_ids):
        with self.graph.local_scope():
            reordered_ids = torch.searchsorted(self.local_to_global_dest, global_ids)
            return local_out[reordered_ids]

    def push_from_remotes(self, local_out, remote_out, global_ids):
        with self.graph.local_scope():
            # global_ids is wrong !!
            reordered_ids = torch.searchsorted(self.local_to_global_dest, global_ids)
            assert(torch.all(self.local_to_global_dest[reordered_ids] == global_ids))
            local_out[reordered_ids] += remote_out
            return local_out

def unit_test_local_bipartite():
    g = BipartiteGraph(torch.tensor([0, 0, 0, 1, 2, 2, 3]), torch.tensor([0, 1, 1, 1, 2, 2, 2]),\
                    torch.device("cuda:0"))
    fsize =32
    inf = torch.ones((g.num_nodes_u,fsize),device = torch.device("cuda:0"),requires_grad = True)
    out = g.gather(inf)
    out.sum().backward()
    out1 = g.self_gather(inf)
    g.pull_from_remotes(out1,torch.tensor([1,2],device = torch.device(0)))
    g.push_from_remotes(out1,torch.ones((1,fsize),device = torch.device(0)),torch.ones([2],device = torch.device(0)))
    print("Test all gradient flows of bipartite graphs.")
    print("This can be a simple wrapper.")
