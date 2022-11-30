import time
import torch
import dgl
import dgl.function as fn
import time
import random
from dgl import heterograph_index
from dgl.utils  import Index
from utils.log import *
dummy_local_graph = dgl.heterograph({('_U', '_E', '_V_local'): ([],[])}, \
                             {'_U': 1, '_V_local': 1})
dummy_remote_graph = dgl.heterograph({('_U', '_E', '_V_remote'): ([],[])}, \
                             {'_U': 1, '_V_remote': 1})
class Bipartite:

    def get_number_of_edges(self):
        return self.indices.shape[0]


 # dgl.heterograph({('a','b','c'):('csc',(torch.tensor([0,2]),torch.tensor([0,1]),torch.tensor([])))},{'a':2,'c':1})
    def __init__(self):
        self.num_in_nodes_local = 0
        self.num_in_nodes_pulled = 0
        self.num_out_local = 0
        self.num_out_remote = 0
        self.self_ids_offset = 0

        self.gpu_id = torch.device(0)
        self.indptr_L = torch.tensor([],dtype = torch.int64)
        self.indices_L = torch.tensor([],dtype = torch.int64)
        self.indptr_R = torch.tensor([],dtype = torch.int64)
        self.indices_R = torch.tensor([],dtype = torch.int64)
        self.out_degrees = torch.tensor([],dtype = torch.int64)

        self.from_ids = {}
        self.push_to_ids = {}
        self.to_offsets = [0]
        self.pull_from_offsets = [0]

        self.graph_local = dummy_local_graph
        self.graph_remote = dummy_remote_graph


    def reconstruct_graph(self,attention = False):
        edge_ids_local= torch.arange(self.indices_L.shape[0], device = self.gpu_id)
        edge_ids_remote = torch.arange(self.indices_R.shape[0], device = self.gpu_id)
        formats = "csc"
        in_nodes = self.num_in_nodes_local + self.num_in_nodes_pulled

        metagraph_index_local = heterograph_index.create_metagraph_index(['_U','_V_local'],[('_U','_E','_V_local')])
        if(self.num_out_local != 0 ):
        # if True:
            hg_local = heterograph_index.create_unitgraph_from_csr(\
                        2,  in_nodes , self.num_out_local, self.indptr_L,
                            self.indices_L, edge_ids_local , formats , transpose = True)
            graph_local = heterograph_index.create_heterograph_from_relations( \
                    metagraph_index_local[0], [hg_local], Index([in_nodes ,self.num_out_local]))
            self.graph_local = dgl.DGLHeteroGraph(graph_local,['_U','_V_local'],['_E'])
            if attention:
                self.graph_local = self.graph_local.formats(['csr','csc','coo'])
                self.graph_local.create_formats_()
        else:
            print("Local graph is none")
            self.graph_local = None

        if self.num_out_remote != 0:
        # if True:
            metagraph_index_remote = heterograph_index.create_metagraph_index\
                    (['_U','_V_remote'],[('_U','_E','_V_remote')])
            hg_remote = heterograph_index.create_unitgraph_from_csr(\
                        2,  in_nodes , self.num_out_remote, self.indptr_R,
                            self.indices_R, edge_ids_remote, formats  , transpose = True)
            graph_remote = heterograph_index.create_heterograph_from_relations( \
                    metagraph_index_remote[0], [hg_remote], Index([in_nodes ,self.num_out_remote]))
            self.graph_remote = dgl.DGLHeteroGraph(graph_remote,['_U','_V_remote'],['_E'])
            # self.graph_remote_csc = self.graph_remote.formats('csr')
            if attention:
                self.graph_remote = self.graph_remote.formats(['csr','csc','coo'])
                self.graph_remote.create_formats_()
        else:
            self.graph_remote = None
            print("remote graph is none")


    def get_from_nds_size(self):
        from_nds_size = {}
        for i in range(4):
            if i != self.gpu_id:
                from_nds_size[i] = self.from_ids[i].shape[0]
        return from_nds_size

    def construct_from_cobject(self, cobject, has_attention= False):
        self.num_in_nodes_local = cobject.num_in_nodes_local
        self.num_in_nodes_pulled = cobject.num_in_nodes_pulled
        self.num_out_local = cobject.num_out_local
        self.num_out_remote = cobject.num_out_remote
        self.self_ids_offset = cobject.self_ids_offset
        self.gpu_id = torch.device(cobject.gpu_id)

        self.indptr_L = cobject.indptr_L
        self.indices_L = cobject.indices_L
        self.indptr_R = cobject.indptr_R
        self.indices_R = cobject.indices_R
        self.out_degrees = cobject.out_degree_local


        self.from_ids = {}
        self.push_to_ids = {}
        self.to_offsets = []
        self.pull_from_offsets = []
        self.to_offsets.append(cobject.to_offsets[0])
        self.pull_from_offsets.append(cobject.to_offsets[0])

        for i in range(4):
            self.from_ids[i] = cobject.from_ids[i]
            self.push_to_ids[i] = cobject.push_to_ids[i]
            self.to_offsets.append(cobject.to_offsets[i+1])
            self.pull_from_offsets.append(cobject.pull_from_offsets[i+1])
            #print("pre serialziation gpu id",self.gpu_id, from_size, to_size)

    def gather_local(self, f_in):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        if self.num_out_local == 0:
            return f_in[0:0,:]
        with self.graph_local.local_scope():
            # FixME Todo: Fix this inconsistency in number of nodes
            # print(f_in.shape[0], self.graph.number_of_nodes('_U'))
            assert(f_in.shape[0] == self.graph_local.number_of_nodes('_U'))
            self.graph_local.nodes['_U'].data['in'] = f_in
            f = self.graph_local.formats()
            self.graph_local.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            assert(f == self.graph_local.formats())
            return self.graph_local.nodes['_V_local'].data['out']

    def gather_remote(self, f_in):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        if self.num_out_remote  == 0:
            return f_in[0:0,:]
        with self.graph_remote.local_scope():
            # FixME Todo: Fix this inconsistency in number of nodes
            assert(f_in.shape[0] == self.graph_remote.number_of_nodes('_U'))
            self.graph_remote.nodes['_U'].data['in'] = f_in
            f = self.graph_remote.formats()
            self.graph_remote.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            assert(f == self.graph_remote.formats())
            return self.graph_remote.nodes['_V_remote'].data['out']

    def gather_local_max(self, nf):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        with self.graph_local.local_scope():
            self.graph_local.edges['_E'].data['nf'] = nf
            self.graph_local.update_all(fn.copy_e('nf', 'm'), fn.max('m', 'out'))
            return self.graph_local.nodes['_V_local'].data['out']

    def gather_remote_max(self, nf):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        if self.num_out_remote == 0:
            return nf[0:0,:]
        with self.graph_remote.local_scope():
            self.graph_remote.edges['_E'].data['nf'] = nf
            self.graph_remote.update_all(fn.copy_e('nf', 'm'), fn.max('m', 'out'))
            return self.graph_remote.nodes['_V_remote'].data['out']

    def slice_owned_nodes(self, out):
        assert(False)

    def self_gather(self,local_in):
        return local_in[:self.self_ids_offset]

    def attention_gather_local(self, attention, u_in):
        with self.graph_local.local_scope():
            self.graph_local.edges['_E'].data['in_e'] = attention
            self.graph_local.nodes['_U'].data['in'] = u_in
            self.graph_local.update_all(fn.u_mul_e(
                'in', 'in_e', 'h'), fn.sum('h', 'out'))
            return self.graph_local.nodes['_V_local'].data['out']

    def attention_gather_remote(self, attention, u_in):
        if self.num_out_remote == 0:
            return u_in[0:0,:]
        with self.graph_remote.local_scope():
            self.graph_remote.edges['_E'].data['in_e'] = attention
            self.graph_remote.nodes['_U'].data['in'] = u_in
            self.graph_remote.update_all(fn.u_mul_e(
                'in', 'in_e', 'h'), fn.sum('h', 'out'))
            return self.graph_remote.nodes['_V_remote'].data['out']

    def pull_dest_remotes(self, local_out, gpu_id):
        assert(local_out.shape[0] == self.num_out_remote )
        with self.graph.local_scope():
            start = self.to_offsets[gpu_id]
            end = self.to_offsets[gpu_id + 1]
            if(end - start  == 0):
                return None
            return local_out[start:end]

    def push_dest_remotes(self, local_out, remote_out, gpu_id):
        with self.graph.local_scope():
            # global_ids is wrong !!
            # Average out these outputs.
            if(self.push_to_ids[gpu_id].shape[0] == 0):
                return None
            local_out[self.push_to_ids[gpu_id]] += remote_out.to(self.gpu_id)
            return local_out


    def apply_edge_local(self, el, er):
        if self.graph_local == None:
            print(el.shape,er.shape)
        with self.graph_local.local_scope():
            self.graph_local.nodes['_V_local'].data['er'] = er
            self.graph_local.nodes['_U'].data['el'] = el
            self.graph_local.apply_edges(fn.u_add_v('el', 'er', 'e'))
            return self.graph_local.edata['e']
    def apply_edge_remote(self, el, er):
        if self.graph_remote == None:
            return  e[0:0,:]
        with self.graph_remote.local_scope():
            self.graph_remote.nodes['_V_remote'].data['er'] = er
            self.graph_remote.nodes['_U'].data['el'] = el
            self.graph_remote.apply_edges(fn.u_add_v('el', 'er', 'e'))
            return self.graph_remote.edata['e']


    def apply_node_local(self, nf):
        with self.graph_local.local_scope():
            self.graph_local.edges['_E'].data['nf'] = nf
            self.graph_local.update_all(fn.copy_e('nf', 'm'), fn.sum('m', 'out'))
            return self.graph_local.nodes['_V_local'].data['out']

    def apply_node_remote(self, nf):
        if self.num_out_remote == 0:
            return nf[0:0,:]
        with self.graph_remote.local_scope():
            self.graph_remote.edges['_E'].data['nf'] = nf
            self.graph_remote.update_all(fn.copy_e('nf', 'm'), fn.sum('m', 'out'))
            return self.graph_remote.nodes['_V_remote'].data['out']

    def copy_from_out_nodes_local(self, local_out):
        # assert(False)
        with self.graph_local.local_scope():
            self.graph_local.nodes['_V_local'].data['out'] = local_out
            self.graph_local.edges['_E'].data['temp'] = torch.zeros(
                    self.graph_local.num_edges('_E'), *local_out.shape[1:], device=self.gpu_id)
            self.graph_local.apply_edges(fn.v_add_e('out', 'temp', 'm'))
            return self.graph_local.edata['m']

    def copy_from_out_nodes_remote(self, local_out):
        # assert(False)
        if self.num_out_remote == 0:
            return local_out[0:0,:]
        with self.graph_remote.local_scope():
            self.graph_remote.nodes['_V_remote'].data['out'] = local_out
            self.graph_remote.edges['_E'].data['temp'] = torch.zeros(
                    self.graph_remote.num_edges('_E'), *local_out.shape[1:], device=self.gpu_id)
            self.graph_remote.apply_edges(fn.v_add_e('out', 'temp', 'm'))
            return self.graph_remote.edata['m']

    def set_remote_data_to_zero(self, data):
        assert(False)
        # Checked that backward pass is not broken.
        # clonedData = data.clone()
        # for i in range(4):
        #     if i != self.gpu_id:
        #         clonedData[self.to_ids[i]] = torch.zeros(
        #             self.to_ids[i].shape[0], *clonedData.shape[1:], device=self.gpu_id)
        # return clonedData
