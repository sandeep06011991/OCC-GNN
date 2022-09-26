import time
import torch
import dgl
import dgl.function as fn
import time
import random
from dgl import heterograph_index
from dgl.utils  import Index
from utils.log import *
class Bipartite:

    def __init__(self):
        self.gpu_id = torch.device(0)
        self.indptr = torch.tensor([],dtype = torch.int64)
        self.expand_indptr = torch.tensor([],dtype = torch.int64)
        self.indices = torch.tensor([],dtype = torch.int64)
        self.N = 0
        self.M = 0
        self.num_nodes_v = 0
        self.from_ids = {}
        self.to_ids = {}
        self.graph = dgl.heterograph({('_V', '_E', '_U'): ([],[])}, \
                                     {'_U': 1, '_V': 1})
        self.in_nodes = torch.tensor([],dtype = torch.int64)
        self.out_nodes = torch.tensor([],dtype = torch.int64)
        self.owned_out_nodes = torch.tensor([],dtype = torch.int64)
        self.in_degree = torch.tensor([],dtype = torch.int64)
        # self.from_size = {i:torch.tensor([],dtype = torch.int32) for i in range(4)}
        # self.to_size = {i:torch.tensor([],dtype = torch.int32) for i in range(4)}
        self.self_ids_in = torch.tensor([],dtype = torch.int64)
        self.self_ids_out = torch.tensor([],dtype = torch.int64)

    def reconstruct_graph(self):
        metagraph_index = heterograph_index.create_metagraph_index(['_U','_V'],[('_V','_E','_U')])
        # # Note dont have to create coo graphs
        # # I can use csr graphs directly
        # But coo will be created in back pass
        # Look for ways to avoid this. 
        hg = heterograph_index.create_unitgraph_from_coo(\
                    2,  self.N, self.M, self.expand_indptr, self.indices, ['coo','csr','csc'])
        graph = heterograph_index.create_heterograph_from_relations( \
                metagraph_index[0], [hg], Index([self.M,self.N]))
        self.graph = dgl.DGLHeteroGraph(graph,['_U','_V'],['_E'])
        self.graph = self.graph.reverse()
        self.graph.create_formats_()
        # self.graph = self.graph.reverse()
        # self.graph = self.graph.formats('csc')
        # self.graph = self.graph.reverse()
        # self.graph = self.graph.formats('csc')
        # self.graph = dgl.heterograph({('_V', '_E', '_U'): (self.expand_indptr.clone(), self.indices.clone())},
        #                              {'_U': self.M, '_V': self.N})
        # self.graph = self.graph.reverse()
        # # print(self.graph.edges(), self.graph)
        # # Check if the graph looks same
        # self.graph_csr = self.graph.formats('csc')
        # self.graph_csc = self.graph.formats('csr')

    def construct_from_cobject(self, cobject, has_attention= False):
        self.gpu_id = torch.device(cobject.gpu_id)
        self.indptr = cobject.indptr
        self.expand_indptr = cobject.expand_indptr
        self.indices = cobject.indices
        if (len(self.indices) != 0):
            N = cobject.num_out_nodes
            M = cobject.num_in_nodes
            self.M = M
            self.N = N
            self.num_nodes_v = N
            assert(self.indptr[-1] == len(self.indices))
            t11 = time.time()
            self.graph = dgl.heterograph({('_V', '_E', '_U'): (self.expand_indptr.clone(), self.indices.clone())},
                                         {'_U': M, '_V': N})
            self.graph = self.graph.reverse()
            self.graph_csr = self.graph.formats('csc')
            self.graph_csc = self.graph.formats('csr')
            assert ['csc' in self.graph.formats()]
        else:
            N = cobject.num_out_nodes
            M = cobject.num_in_nodes
            self.M = M
            self.N = N
            self.num_nodes_v = N
            # print("graph created attempt", N, M)
            self.graph = dgl.heterograph({('_V', '_E', '_U'): ([],[])}, \
                                         {'_U': M, '_V': N})
            self.graph = self.graph.reverse()
            self.graph = self.graph.formats('csc')
        self.in_nodes = cobject.in_nodes
        self.out_nodes = cobject.out_nodes
        self.owned_out_nodes = cobject.owned_out_nodes
        self.owned_out_nodes = cobject.owned_out_nodes
        self.in_nodes = cobject.in_nodes
        self.out_nodes = cobject.out_nodes
        self.owned_out_nodes = cobject.owned_out_nodes
        self.in_degree = cobject.indegree
        self.from_ids = {}
        self.to_ids = {}

        from_size = {}
        to_size = {}
        for i in range(4):
            self.from_ids[i] = cobject.from_ids[i]
            self.to_ids[i] = cobject.to_ids[i]
            #print("pre serialziation gpu id",self.gpu_id, from_size, to_size)
        self.self_ids_in = cobject.self_ids_in
        self.self_ids_out = cobject.self_ids_out
        self.owned_out_nodes =cobject.owned_out_nodes

    def gather(self, f_in):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        if self.num_nodes_v == 0:
            assert(False)
            # Might cause a silent failure.
            return f_in
        with self.graph.local_scope():
            # FixME Todo: Fix this inconsistency in number of nodes
            print(f_in.shape[0], self.graph.number_of_nodes('_U'))
            assert(f_in.shape[0] == self.graph.number_of_nodes('_U'))
            self.graph.nodes['_U'].data['in'] = f_in
            f = self.graph.formats()
            self.graph.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            # No new formats must be created.
            if(f != self.graph.formats()):
                LogFile("bipartite", 1).log("Created new graph formats from {} to {}".format(f, self.graph.formats()))
                print("Created new graph formats from {} to {}".format(f, self.graph.formats()))
            # assert(f == self.graph.formats())
            # print(self.graph.nodes['_V'].data['out'])
            return self.graph.nodes['_V'].data['out']

    def slice_owned_nodes(self, f_in):
        with self.graph.local_scope():
            if self.owned_out_nodes.shape[0] == 0:
                return f_in[0:0,:]
            return f_in[self.owned_out_nodes]

    def attention_gather(self, attention, u_in):
        with self.graph.local_scope():
            self.graph.edges['_E'].data['in_e'] = attention
            self.graph.nodes['_U'].data['in'] = u_in
            self.graph.update_all(fn.u_mul_e(
                'in', 'in_e', 'h'), fn.sum('h', 'out'))
            return self.graph.nodes['_V'].data['out']

    def self_gather(self, local_in):
        with self.graph.local_scope():
            out = torch.zeros(self.num_nodes_v,
                              *local_in.shape[1:], device=self.gpu_id)
            out[self.self_ids_out] = local_in[self.self_ids_in]
            return out

    def pull_for_remotes(self, local_out, gpu_id):
        with self.graph.local_scope():
            if(self.to_ids[gpu_id].shape[0] == 0):
                return None
            return local_out[self.to_ids[gpu_id]]

    def push_from_remotes(self, local_out, remote_out, gpu_id):
        with self.graph.local_scope():
            # global_ids is wrong !!
            # Average out these outputs.
            if(self.from_ids[gpu_id].shape[0] == 0):
                return None
            local_out[self.from_ids[gpu_id]] += remote_out.to(self.gpu_id)
            return local_out


    def apply_edge(self, el, er):
        with self.graph.local_scope():
            self.graph.nodes['_V'].data['er'] = er
            self.graph.nodes['_U'].data['el'] = el
            self.graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            return self.graph.edata['e']

    def apply_node(self, nf):
        with self.graph.local_scope():
            self.graph.edges['_E'].data['nf'] = nf
            self.graph.update_all(fn.copy_e('nf', 'm'), fn.sum('m', 'out'))
            return self.graph.nodes['_V'].data['out']

    def copy_from_out_nodes(self, local_out):
        with self.graph.local_scope():
            self.graph.nodes['_V'].data['out'] = local_out
            self.graph.edges['_E'].data['temp'] = torch.zeros(
                self.graph.num_edges('_E'), local_out[1].shape[1], device=self.gpu_id)
            self.graph.apply_edges(fn.v_add_e('out', 'temp', 'm'))
            return self.graph.edata['m']

    def set_remote_data_to_zero(self, data):
        # Checked that backward pass is not broken.
        clonedData = data.clone()
        for i in range(4):
            if i != self.gpu_id:
                clonedData[self.to_ids[i]] = torch.zeros(
                    self.to_ids[i].shape[0], *clonedData.shape[1:], device=self.gpu_id)
        return clonedData
