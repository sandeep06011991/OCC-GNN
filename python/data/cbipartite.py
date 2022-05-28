
# Performs all tensorization and equivalent transformations.
# csr_matrix((data, indices, indptr), [shape=(M, N)])
# is the standard CSR representation where the column indices for row i
# are stored in indices[indptr[i]:indptr[i+1]]
# and their corresponding values are stored in data[indptr[i]:indptr[i+1]].
# If the shape parameter is not supplied, the matrix dimensions
# are inferred from the index arrays.

import numpy as np
import scipy as sp
import torch
import dgl
import torch
import dgl.function as fn
import time

class Bipartite:

    def __init__(self, cobject):
        self.gpu_id = torch.device(cobject.gpu_id)
        indptr = cobject.indptr
        indices = cobject.indices
        t1 = time.time()
        if(len(indices)!=0):
            N =  cobject.num_out_nodes
            M =  cobject.num_in_nodes
            self.num_nodes_v = N
            #print("graph created attempt", N, M)
            #print(indices[-1])
            assert(indptr[-1] == len(indices))
            '''sp_mat = sp.sparse.csr_matrix((np.ones(indices.shape),\
                    indices.numpy(), indptr.numpy()), \
                        shape = (N,M))
            self.graph = dgl.bipartite_from_scipy(sp_mat,  utype='_V', \
                                                etype='_E', vtype='_U' ,device = self.gpu_id)
            # Had to reverse graph to match sampling and conv directions
            self.graph = self.graph.reverse()
            '''
            self.graph = dgl.heterograph({('_U','_E','_V'): \
                        (cobject.indices, cobject.expand_indptr)},\
                        {'_U': M, '_V': N}, device = self.gpu_id)
            #print("graph created", N, M)
        else:
            self.num_nodes_v = 0
            # empty code such that other things dont break.
            self.graph = dgl.graph([])
        t2 = time.time()
        self.in_nodes = cobject.in_nodes.to(device = self.gpu_id)
        self.out_nodes = cobject.out_nodes.to(device = self.gpu_id)
        self.owned_out_nodes = cobject.owned_out_nodes.to(device = self.gpu_id)

        from_ids = []
        for i in range(4):
            from_ids.append(cobject.from_ids[i].to(device = self.gpu_id))
        self.from_ids = from_ids

        to_ids = []
        for i in range(4):
            to_ids.append(cobject.to_ids[i].to(device = self.gpu_id))
        self.to_ids = to_ids
        self.self_ids_in = cobject.self_ids_in.to(device = self.gpu_id)
        self.self_ids_out = cobject.self_ids_out.to(device = self.gpu_id)
        t3 = time.time()
        print("Graph construction time ",t2-t1)
        print("tensorize everything", t3 - t2)

    def gather(self, f_in):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        with self.graph.local_scope():
            # print(f_in.shape[0], self.graph.number_of_nodes('_U'))
            # FixME Todo: Fix this inconsistency in number of nodes
            assert(f_in.shape[0] ==  self.graph.number_of_nodes('_U'))
            self.graph.nodes['_U'].data['in'] = f_in
            self.graph.update_all(fn.copy_u('in', 'm'), fn.mean('m', 'out'))
            return self.graph.nodes['_V'].data['out']

    def slice_owned_nodes(self,f_in):
        with self.graph.local_scope():
            if self.owned_out_nodes.shape[0]==0:
                return torch.zeros(0,f_in.shape[1],device = self.gpu_id)
            return f_in[self.owned_out_nodes]

    def attention_gather(self, v_in, u_in):
        with self.graph.local_scope():
            self.graph.nodes['_V'].data['in_v'] = v_in
            self.graph.nodes['_U'].data['in'] = u_in
            self.graph.update_all(fn.u_mul_v('in','in_v','h'),fn.sum('h','out'))
            return self.graph.nodes['_V'].data['out']

    def self_gather(self, local_in):
        with self.graph.local_scope():
            out = torch.zeros(self.num_nodes_v,local_in.shape[1],device =  self.gpu_id)
            out[self.self_ids_out] = local_in[self.self_ids_in]
            return out

    def pull_for_remotes(self,local_out, gpu_id):
        with self.graph.local_scope():
            if(self.to_ids[gpu_id].shape[0]==0):
                return None
            return local_out[self.to_ids[gpu_id]]

    def push_from_remotes(self, local_out, remote_out, gpu_id):
        with self.graph.local_scope():
            # global_ids is wrong !!
            # Average out these outputs.
            if(self.from_ids[gpu_id].shape[0]==0):
                return None
            local_out[self.from_ids[gpu_id]] += remote_out.to(self.gpu_id)
            return local_out

class Sample:
    def __init__(self, csample):
        self.in_nodes = csample.in_nodes
        self.out_nodes = csample.out_nodes
        self.layers = []
        # print(len(csample.layers))
        for layer in csample.layers:
            l = []
            # print(len(layer))
            for cbipartite in layer:
                l.append(Bipartite(cbipartite))
            self.layers.append(l)

        self.last_layer_nodes = []
        last_layer = self.layers[0]
        i = 0
        for l in last_layer:
             assert(torch.device(i) == l.gpu_id )
             self.last_layer_nodes.append(l.out_nodes[l.owned_out_nodes])
             i= i+1
        self.layers.reverse()
        #print("Sample creation complete")
