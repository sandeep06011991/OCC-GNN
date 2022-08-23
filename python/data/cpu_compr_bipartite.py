
# Performs all tensorization and equivalent transformations.
# csr_matrix((data, indices, indptr), [shape=(M, N)])
# is the standard CSR representation where the column indices for row i
# are stored in indices[indptr[i]:indptr[i+1]]
# and their corresponding values are stored in data[indptr[i]:indptr[i+1]].
# If the shape parameter is not supplied, the matrix dimensions
# are inferred from the index arrays.

from importlib.metadata import metadata
import numpy as np
import scipy as sp
import torch
import dgl
import torch
import dgl.function as fn
import time
import random

class Bipartite:

    def serialize(self):
        metadatalist = [
            self.indptr_start,
            self.indptr_end,
            self.expand_indptr_start,
            self.expand_indptr_end,
            self.indices_start,
            self.indices_end,
            self.in_nodes_start,
            self.in_nodes_end,
            self.out_nodes_start,
            self.out_nodes_end,
            self.owned_out_nodes_start,
            self.owned_out_nodes_end,
            self.self_ids_in_start,
            self.self_ids_in_end,
            self.self_ids_out_start,
            self.self_ids_out_end,
            self.indegree_start,
            self.indegree_end
        ]

        def listFromObj(obj):
            return [obj[key] for key in range(4)]
        mark = len(metadatalist)
        metadatalist.extend(listFromObj(self.from_ids_start))
        metadatalist.extend(listFromObj(self.from_ids_end))
        metadatalist.extend(listFromObj(self.to_ids_start))
        metadatalist.extend(listFromObj(self.to_ids_end))
        #print("pre serialize calculate",from_d , to_d)
        assert(len(metadatalist)==34)
        tensor = torch.tensor(metadatalist, dtype=torch.long)
        tensorCatData = torch.cat([tensor, self.data])
        checksum = torch.sum(tensor)
        return (tensorCatData, self.graph, self.gpu_id, checksum)

    @staticmethod
    def deserialize(data):
        tensor, graph, gpu_id, checksum = data
        metadatalist = tensor[:34].tolist()
        #print("post serialize calculate",from_d, to_d)
        #print("Check sum", checksum, sum(metadatalist))
        data = tensor[34:]
        bipartite = Bipartite(metadatalist=metadatalist,
                              data=data, graph=graph, gpu_id=gpu_id)
        return bipartite

    def __init__(self, cobject=None,  metadatalist=None, data=None, graph=None, gpu_id=None):
        if metadatalist is not None and data is not None and graph is not None and gpu_id is not None:
            self.graph = graph
            # self.graph = self.graph.formats(['csr', 'coo', 'csc'])
            # self.graph.create_formats_()
            self.gpu_id = gpu_id

            self.indptr_start = metadatalist[0]
            self.indptr_end = metadatalist[1]
            self.expand_indptr_start = metadatalist[2]
            self.expand_indptr_end = metadatalist[3]
            self.indices_start = metadatalist[4]
            self.indices_end = metadatalist[5]
            self.in_nodes_start = metadatalist[6]
            self.in_nodes_end = metadatalist[7]
            self.out_nodes_start = metadatalist[8]
            self.out_nodes_end = metadatalist[9]
            self.owned_out_nodes_start = metadatalist[10]
            self.owned_out_nodes_end = metadatalist[11]
            self.self_ids_in_start = metadatalist[12]
            self.self_ids_in_end = metadatalist[13]
            self.self_ids_out_start = metadatalist[14]
            self.self_ids_out_end = metadatalist[15]
            self.indegree_start = metadatalist[16]
            self.indegree_end = metadatalist[17]
            self.num_nodes_v = self.out_nodes_end - self.out_nodes_start
            self.from_ids_start = {}
            self.from_ids_end = {}
            self.to_ids_start = {}
            self.to_ids_end = {}
            from_size = {}
            to_size = {}
            for i in range(4):
                self.from_ids_start[i] = metadatalist[18 + i]
                self.from_ids_end[i] = metadatalist[22 + i]
                self.to_ids_start[i] = metadatalist[26 + i]
                self.to_ids_end[i] = metadatalist[30 + i]
                from_size[i] = self.from_ids_end[i] - self.from_ids_start[i]
                to_size[i] = self.to_ids_end[i] - self.to_ids_start[i]
            #print("post serialization gpu id",self.gpu_id, from_size, to_size)

            self.data = data
            return

        # Not From Serialization
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
        if(len(indices) != 0):
            N = cobject.num_out_nodes
            M = cobject.num_in_nodes
            self.num_nodes_v = N
            assert(indptr[-1] == len(indices))
            t11 = time.time()
            self.graph = dgl.heterograph({('_V', '_E', '_U'): (expand_indptr.clone(), indices.clone())},
                                         {'_U': M, '_V': N})
            self.graph = self.graph.reverse()
            self.graph = self.graph.formats('csc')
            assert ['csc' in self.graph.formats()]
        else:
            N = cobject.num_out_nodes
            M = cobject.num_in_nodes
            self.num_nodes_v = N
            # print("graph created attempt", N, M)
            self.graph = dgl.heterograph({('_V', '_E', '_U'): ([],[])}, \
                                         {'_U': M, '_V': N})
            self.graph = self.graph.reverse()
            self.graph = self.graph.formats('csc')
            # empty code such that other things dont break.
            # self.graph = dgl.graph([])
        t2 = time.time()
        self.in_nodes_start = cobject.in_nodes_start
        self.in_nodes_end = cobject.in_nodes_end
        self.out_nodes_start = cobject.out_nodes_start
        self.out_nodes_end = cobject.out_nodes_end
        self.owned_out_nodes_start = cobject.owned_out_nodes_start
        self.owned_out_nodes_end = cobject.owned_out_nodes_end
        self.indegree_start = cobject.indegree_start
        self.indegree_end = cobject.indegree_end
        # self.in_nodes = data[cobject.in_nodes_start:cobject.in_nodes_end]
        # self.out_nodes = data[cobject.out_nodes_start: cobject.out_nodes_end]
        # self.owned_out_nodes = data[cobject.owned_out_nodes_start:cobject.owned_out_nodes_end]
        self.from_ids_start = {}
        self.from_ids_end = {}
        self.to_ids_start = {}
        self.to_ids_end = {}
        from_size = {}
        to_size = {}
        for i in range(4):
            self.from_ids_start[i] = cobject.from_ids_start[i]
            self.from_ids_end[i] = cobject.from_ids_end[i]
            self.to_ids_start[i] = cobject.to_ids_start[i]
            self.to_ids_end[i] = cobject.to_ids_end[i]
            from_size[i] = self.from_ids_end[i] - self.from_ids_start[i]
            to_size[i] = self.to_ids_end[i] - self.to_ids_start[i]
        #print("pre serialziation gpu id",self.gpu_id, from_size, to_size)    
        self.self_ids_in_start = cobject.self_ids_in_start
        self.self_ids_in_end = cobject.self_ids_in_end
        self.self_ids_out_start = cobject.self_ids_out_start
        self.self_ids_out_end = cobject.self_ids_out_end
        from_ids = {}
        self.owned_out_nodes = data[cobject.owned_out_nodes_start: cobject.owned_out_nodes_end]
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
        self.data = self.data.to(self.gpu_id)
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
        self.in_degree = data[self.indegree_start:self.indegree_end]
        self.in_degree = self.in_degree.reshape(self.in_degree.shape[0],1)
        
        t3 = time.time()
    #
    def debug(self):
        print("gpu id",self.gpu_id)
        #print("Graph", self.graph)
        #print("in nodes",self.in_nodes.shape)
        #print("out nodes",self.out_nodes)
        #print("owned_out_nodes",self.owned_out_nodes)
        #print("from_ids",self.from_ids)
        to = {}
        fr = {}
        for i in self.from_ids.keys():
            to[i] = self.to_ids[i].shape[0]
            fr[i] = self.from_ids[i].shape[0]
        print("to",to,"from",fr)    
        #print("to ids",self.to_ids)
        #print("self ids in",self.self_ids_in)
        #print("self ids out",self.self_ids_in)

    def gather(self, f_in):
        #print(f_in.shape, self.graph.number_of_nodes('_U'))
        if self.num_nodes_v == 0:
            return f_in
        with self.graph.local_scope():
            # print(f_in.shape[0], self.graph.number_of_nodes('_U'))
            # FixME Todo: Fix this inconsistency in number of nodes
            # print(f_in.shape, self.graph.number_of_nodes('_U'))
            assert(f_in.shape[0] == self.graph.number_of_nodes('_U'))
            self.graph.nodes['_U'].data['in'] = f_in
            f = self.graph.formats()
            self.graph.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            # No new formats must be created.
            assert(f == self.graph.formats())
            # print(self.graph.nodes['_V'].data['out'])
            return self.graph.nodes['_V'].data['out']

    def slice_owned_nodes(self, f_in):
        with self.graph.local_scope():
            if self.owned_out_nodes.shape[0] == 0:
                return f_in[0:0,:]
                # Horrible bug caused autograd to fail
                # return torch.zeros(0, f_in.shape[1], device=self.gpu_id)
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
                              local_in.shape[1], device=self.gpu_id)
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
        clonedData = data.clone()
        for i in range(4):
            if i != self.gpu_id:
                clonedData[self.to_ids[i]] = torch.zeros(
                    self.to_ids[i].shape[0], *clonedData.shape[1:], device=self.gpu_id)
        return clonedData


class Sample:
    def __init__(self, csample):
        self.in_nodes = csample.in_nodes
        self.out_nodes = csample.out_nodes
        self.layers = []
        
        self.randid = random.randint(0,10000)
        # print(len(csample.layers))
        for layer in csample.layers:
            l = []
            # print(len(layer))
            for cbipartite in layer:
                l.append(Bipartite(cbipartite))
            self.layers.append(l)


class Gpu_Local_Sample:
    def __init__(self, global_sample=None, device_id=None, randid = None,  in_nodes=None, out_nodes=None, serializedLayers=None):
        if in_nodes is not None and out_nodes is not None and serializedLayers is not None and device_id is not None:
            self.in_nodes = in_nodes
            self.out_nodes = out_nodes
            self.randid = randid
            self.layers = []
            for layer in serializedLayers:
                self.layers.append(Bipartite.deserialize(layer))
            self.device_id = device_id
            return

        # print(global_sample, in_nodes, out_nodes, serializedLayers, device_id,
            #   in_nodes is None and out_nodes is not None and serializedLayers is not None and device_id is not None)
        self.in_nodes = global_sample.in_nodes
        self.out_nodes = global_sample.out_nodes
        self.randid = global_sample.randid
        self.layers = []
        for layer in global_sample.layers:
            self.layers.append(layer[device_id])
        # self.last_layer_nodes = global_sample.last_layer_nodes[device_id]
        self.device_id = device_id

    def serialize(self):
        serializedLayers = tuple([i.serialize() for i in self.layers])
        return (self.in_nodes, self.out_nodes, self.randid, serializedLayers, self.device_id)

    @staticmethod
    def deserialize(tensor):
        in_nodes, out_nodes, randid , serializedLayers, device_id = tensor
        return Gpu_Local_Sample(device_id=device_id, in_nodes=in_nodes, \
                        randid = randid, out_nodes=out_nodes, serializedLayers=serializedLayers)

    def __str__(self):
        return "TEST: " + str(self.serialize())

    # Init from a serialized
    # @classmethod
    # def __init__(self, in_nodes, out_nodes, serializedLayers, device_id):
    #     self.in_nodes = in_nodes
    #     self.out_nodes = out_nodes
    #     self.layers = []
    #     for layer in serializedLayers:
    #         self.layers.append(Bipartite.deserializeFromTensor(layer))
    #     self.device_id = device_id

    def debug(self):
        print("Working on ",self.randid, "gpu", self.device_id)
        #for id,l in enumerate(self.layers):
        #    print("layer info ",id)
        #    l.debug()
        #print("last layer nodes", self.last_layer_nodes.shape)

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
        return dgl.merge(tt), torch.cat(dd, dim=0)

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
        self.last_layer_nodes = l.out_nodes[l.owned_out_nodes].to(
            self.device_id)
        self.layers.reverse()
