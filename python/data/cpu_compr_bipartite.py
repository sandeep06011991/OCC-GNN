
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
        self.data = cobject.data_tensor.clone()
        data = self.data
        data.share_memory_()
        self.indptr_start = cobject.indptr_start
        self.indptr_end = cobject.indptr_end
        self.expand_indptr_start = cobject.expand_indptr_start
        self.expand_indptr_end = cobject.expand_indptr_end
        self.indices_start = cobject.indices_start
        self.indices_end = cobject.indices_end
        indptr = data[cobject.indptr_start:cobject.indptr_end]
        expand_indptr = data[cobject.expand_indptr_start: cobject.expand_indptr_end]
        indices = data[cobject.indices_start:cobject.indices_end]
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
            t11 = time.time()
            self.graph = dgl.heterograph({('_V','_E','_U'):(expand_indptr.clone(),indices.clone())}, \
                            {'_U':M, '_V':N})
            self.graph = self.graph.reverse()
            self.graph.create_formats_()
            t22 = time.time()
            g1 = dgl.heterograph({('_U','_E','_V'):('csc',self.graph.adj_sparse('csc'))},{'_U':M, '_V':N})
            t33 = time.time()
            print("Check alternative",t22 -t11)
            print("From Tensor again",t33 - t22)

            assert ['csc' in self.graph.formats()]

            # i1,i2,i3 = self.graph.adj_sparse('csr')
            # assert(torch.all(i1==indptr))
            # assert(torch.all(i2 == indices))
            # print("All ok !")
            # self.graph = dgl.heterograph({('_V','_E','_U'): \
            #             ('csr',( indptr, indices ,[]))},\
            #             {'_U': M, '_V': N}, device = self.gpu_id)
            # ind,indices,edges = self.graph.adj_sparse('csr')
            # assert(ind.shape[0] == M+1 )
            # assert(torch.all(indices < N))
            #print("graph created", N, M)
        else:
            self.num_nodes_v = 0
            # empty code such that other things dont break.
            # self.graph = dgl.graph([])
        t2 = time.time()
        self.in_nodes_start = cobject.in_nodes_start
        self.in_nodes_end = cobject.in_nodes_end
        self.out_nodes_start = cobject.out_nodes_start
        self.out_nodes_end = cobject.out_nodes_end
        self.owned_out_nodes_start = cobject.owned_out_nodes_start
        self.owned_out_nodes_end = cobject.owned_out_nodes_end
        # self.in_nodes = data[cobject.in_nodes_start:cobject.in_nodes_end]
        # self.out_nodes = data[cobject.out_nodes_start: cobject.out_nodes_end]
        # self.owned_out_nodes = data[cobject.owned_out_nodes_start:cobject.owned_out_nodes_end]
        self.from_ids_start = {}
        self.from_ids_end = {}
        self.to_ids_start = {}
        self.to_ids_end = {}
        for i in range(4):
            self.from_ids_start[i] = cobject.from_ids_start[i]
            self.from_ids_end[i] = cobject.from_ids_end[i]
            self.to_ids_start[i] = cobject.to_ids_start[i]
            self.to_ids_end[i] = cobject.to_ids_end[i]
        self.self_ids_in_start = cobject.self_ids_in_start
        self.self_ids_in_end = cobject.self_ids_in_end
        self.self_ids_out_start = cobject.self_ids_out_start
        self.self_ids_out_end = cobject.self_ids_out_end
        # from_ids = {}
        # for i in range(4):
        #     from_ids[i] = (data[cobject.from_ids_start[i]: cobject.from_ids_end[i]])
        #     if i == self.gpu_id:
        #         print(from_ids[i])
        # self.from_ids = from_ids
        #
        # to_ids = {}
        # for i in range(4):
        #     to_ids[i] = (data[cobject.to_ids_start[i]:cobject.to_ids_end[i]])
        #     if i == self.gpu_id:
        #         print(from_ids[i])
        # self.to_ids = to_ids
        # self.self_ids_in = data[cobject.self_ids_in_start:cobject.self_ids_in_end]
        # self.self_ids_out = data[cobject.self_ids_out_start: cobject.self_ids_out_end]
        # t3 = time.time()
        # print("Graph construction time ",t2-t1)
        # print("tensorize everything", t3 - t2)
        '''data_moved = self.in_nodes.shape[0] + self.out_nodes.shape[0] + self.owned_out_nodes.shape[0]
        for i in range(4):
            data_moved += self.to_ids[i].shape[0] + self.from_ids[i].shape[0]
        data_moved += self.self_ids_in.shape[0] + self.self_ids_out.shape[0]
        data_in_GB = data_moved * 4/ (1024 * 1024 * 1024)
        dummy = torch.rand(data_moved)
        t11 = time.time()
        dummy.to(self.gpu_id)
        t22 = time.time()
        print("bandwidth {} GBps data size {} ishape".format(data_in_GB/(t22-t11), data_moved))
        print("bnandwidth {} GBps data size{}GB".format(data_in_GB/(t3-t2), data_in_GB))'''

    def get_size(self):
        return self.data.shape[0]
    def to_gpu(self):
        self.data  = self.data.to(self.gpu_id)
        data = self.data
        self.graph = self.graph.to(self.gpu_id)
        assert ['csc' in self.graph.formats()]
        self.in_nodes = data[self.in_nodes_start:self.in_nodes_end]
        self.out_nodes = data[self.out_nodes_start: self.out_nodes_end]
        self.owned_out_nodes = data[self.owned_out_nodes_start:self.owned_out_nodes_end]

        from_ids = {}
        for i in range(4):
            from_ids[i] = (data[self.from_ids_start[i]: self.from_ids_end[i]])
            if i == self.gpu_id:
                print(from_ids[i])
        self.from_ids = from_ids

        to_ids = {}
        for i in range(4):
            to_ids[i] = (data[self.to_ids_start[i]:self.to_ids_end[i]])
            if i == self.gpu_id:
                print(from_ids[i])
        self.to_ids = to_ids
        self.self_ids_in = data[self.self_ids_in_start:self.self_ids_in_end]
        self.self_ids_out = data[self.self_ids_out_start: self.self_ids_out_end]
        t3 = time.time()
    #
    # def debug(self):
    #     print("self data shape",self.data.shape)
    def gather(self, f_in):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        with self.graph.local_scope():
            # print(f_in.shape[0], self.graph.number_of_nodes('_U'))
            # FixME Todo: Fix this inconsistency in number of nodes
            assert(f_in.shape[0] ==  self.graph.number_of_nodes('_U'))
            self.graph.nodes['_U'].data['in'] = f_in
            f = self.graph.formats()
            self.graph.update_all(fn.copy_u('in', 'm'), fn.mean('m', 'out'))
            # No new formats must be created.
            assert(f == self.graph.formats())
            # print(self.graph.nodes['_V'].data['out'])
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

        # self.last_layer_nodes = []
        # last_layer = self.layers[0]
        # i = 0
        # for l in last_layer:
        #      assert(torch.device(i) == l.gpu_id )
        #      self.last_layer_nodes.append(l.out_nodes[l.owned_out_nodes])
        #      i= i+1
        # self.layers.reverse()
        #print("Sample creation complete")

class Gpu_Local_Sample:
    def __init__(self, global_sample,device_id):
        self.in_nodes = global_sample.in_nodes
        self.out_nodes = global_sample.out_nodes
        self.layers = []
        for layer in global_sample.layers:
            self.layers.append(layer[device_id])
        # self.last_layer_nodes = global_sample.last_layer_nodes[device_id]
        self.device_id = device_id

    def debug(self):
        for l in self.layers:
            l.debug()
        print("last layer nodes",self.last_layer_nodes.shape)

    def get_tensor(self):
        tt = []
        dd = []
        for l in self.layers:
            g = l.graph.clone()
            g.create_formats_()
            dd.append(l.data)
            tt.append(g)
        # torch.cat(dd,dim=0) -- .001 overhead while fp is .01
        # dgl.merge(tt) 0.002492189407348633
        # tt .06
        # dd .01
        return dgl.merge(tt), torch.cat(dd,dim=0) 

    def get_size(self):
        s = 0
        for l in self.layers:
            s += l.get_size()
        return s
    def to_gpu(self):
        for l in self.layers:
            l.to_gpu()
        self.last_layer_nodes = []
        last_layer = self.layers[0]
        i = 0
        l = self.layers[0]
        self.last_layer_nodes =  l.out_nodes[l.owned_out_nodes].to(self.device_id)
        self.layers.reverse()
