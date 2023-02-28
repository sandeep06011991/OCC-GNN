import dgl
import torch
import dgl.function as fn
from dgl import heterograph_index
from dgl.utils  import Index
in_nodes = 8
num_out_nodes = 2
metagraph_index_local = heterograph_index.\
    create_metagraph_index(['_U','_V_local'],[('_U','_E','_V_local')])

hg_local = heterograph_index.create_unitgraph_from_csr(\
            2,  in_nodes , num_out_nodes, torch.tensor([0,1,6], dtype = torch.int64, device = 0),
                torch.tensor([2,6,5,3,2,1], dtype = torch.int64, device = 0), \
                torch.tensor([0,1,2,3,4,6], dtype = torch.int64, device = 0) ,\
                 "csc", transpose = True)
graph_local = heterograph_index.create_heterograph_from_relations( \
        metagraph_index_local[0], [hg_local], Index([in_nodes ,num_out_nodes]))
graph_local = dgl.DGLHeteroGraph(graph_local,['_U','_V_local'],['_E'])
graph_local.nodes['_U'].data['in'] = torch.rand((8,3),requires_grad = True, device = 0)
torch.cuda.synchronize()
print(graph_local.nodes['_U'].data['in'] )
graph_local.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
print(graph_local.nodes['_V_local'].data['out'])
print(graph_local.nodes['_V_local'].data['out'].sum().backward())
print("All ok")
