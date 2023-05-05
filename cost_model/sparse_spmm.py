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
import torch_scatter
from torch_scatter import segment_csr
import torch_sparse
import torch_geometric




import torch
import torch_geometric
import torch_sparse
def run(N, E, FSize):
    indptr = torch.arange(N+1, device = 0) * 20
    indices = torch.randint(N,(E,), device = 0)
    f_in =  torch.rand(N, 128, device = 0, requires_grad = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    for _ in range(10):
        e1.record()
        spg_local = torch_sparse.SparseTensor(rowptr = indptr, col = indices, sparse_sizes = (N, N) , is_sorted = False, trust_data = True) 
        e2.record()
        e2.synchronize()
        print("torch geo creation elapsed time", e1.elapsed_time(e2)/1000)
    for i in range(10):
        e1.record()
        o =  torch_geometric.utils.spmm(spg_local, f_in)
        e2.record()
        o.sum().backward()
        print(o)
        e3.record()
        e3.synchronize()
        print("torch geo spmm forward elapsed time", e1.elapsed_time(e2)/1000)
        print("torch geo spmm backward elapsed time", e2.elapsed_time(e3)/1000)



def run_performance(N, E, FSize):
    indptr = torch.arange(N+1, device = 0) * 20
    indices = torch.randint(N,(E,), device = 0)
    f_in =  torch.rand(N, 128, device = 0, requires_grad = True)
    edge_ids_local= torch.arange(indices.shape[0], device = 0)
    in_nodes = N
    num_out_local = N
    
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    formats = "csc"
    
    for _ in range(1):
        e1.record()
        metagraph_index_local = heterograph_index.create_metagraph_index(['_U','_V_local'],[('_U','_E','_V_local')])
        hg_local = heterograph_index.create_unitgraph_from_csr(\
                        2,  in_nodes , num_out_local, indptr,
                            indices, edge_ids_local , formats , transpose = True)
        graph_local = heterograph_index.create_heterograph_from_relations( \
                    metagraph_index_local[0], [hg_local], Index([in_nodes ,num_out_local]))
        graph_local = dgl.DGLHeteroGraph(graph_local,['_U','_V_local'],['_E'])
        e2.record()
        e2.synchronize()
        print("Creationg time", e1.elapsed_time(e2)/1000)
    s1 = torch.cuda.Stream()
    d = dgl._ffi.streams.StreamContext(s1)

    torch.cuda.set_stream(s1)
    with graph_local.local_scope():
            # FixME Todo: Fix this inconsistency in number of nodes
            # print(f_in.shape[0], self.graph_local.number_of_nodes('_U'))
        assert(f_in.shape[0] == graph_local.number_of_nodes('_U'))
        graph_local.nodes['_U'].data['in'] = f_in
        f = graph_local.formats()
        for _ in range(10):
            #e1.record()
            assert(f_in.shape[0] == graph_local.number_of_nodes('_U'))
            graph_local.nodes['_U'].data['in'] = f_in
            f = graph_local.formats()
            #with torch.cuda.stream(s1):
            with d:
                graph_local.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
            #e2.record()
                graph_local.nodes['_V_local'].data['out'].sum().backward()
            assert(f == graph_local.formats())
            #e3.record()
            #e3.synchronize()
            #print("DGL forward runtime is ", e1.elapsed_time(e2)/1000)
            #print("DGL Backward time is ", e2.elapsed_time(e3)/1000)
        assert(f == graph_local.formats())
        return graph_local.nodes['_V_local'].data['out']




if __name__ == "__main__":
    N = 5000
    E = 37000
    #run(N , E, 128)
    #assert(False)
    #run_performance(N, E, 128)
    N = 500000
    E = 3600000
    #run(N, E, 128)
    run_performance(N, E, 512)
    #E = 2900000
    #run_performance(N, E, 512)
