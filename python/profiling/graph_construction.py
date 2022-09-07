import dgl
import numpy as np
import time
import torch
from dgl import heterograph_index
from dgl.utils  import Index
from dgl import function as fn
N = 10000
e1 = torch.zeros((N,),dtype = torch.int64)
e2 = torch.arange(0,N)
print(e1,e2)

F = torch.rand(N,100)
t1 = time.time()
g0 = dgl.heterograph({('_V','_E','_U'):(e1,e2)},{'_U':N,'_V':1})
g0 = g0.reverse()
g0.nodes['_U'].data['in'] = F
g0.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
result = g0.nodes['_V'].data['out']
t2 = time.time()
print("SUM BASE",torch.sum(result))


num_src = N
num_dst = 1
t3 = time.time()
arrays = [e1,e2]
metagraph_index = heterograph_index.create_metagraph_index(['_U','_V'],[('_V','_E','_U')])
hg = heterograph_index.create_unitgraph_from_coo(\
                    2,  num_dst, num_src, arrays[0], arrays[1], ['coo'])
graph = heterograph_index.create_heterograph_from_relations(metagraph_index[0], [hg], Index([N,1]))
v = dgl.DGLHeteroGraph(graph,['_U','_V'],['_E'])
t4 = time.time()
g1 = v.reverse()
t5 = time.time()

g1.nodes['_U'].data['in'] = F
g1.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
result = g1.nodes['_V'].data['out']
print("TEST", torch.sum(result))
print("Base construction",t2-t1, t4 - t3, t5-t4)
