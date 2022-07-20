import dgl
import dgl.function as fn
import numpy as np
import scipy as sp
import torch
#
indptr = np.array([0,3])
indices = np.array([0,1,2])
values = np.array([1,1,1])
matrix  = sp.sparse.csr_matrix((values,  indices , indptr))
graph1 = dgl.bipartite_from_scipy(matrix , utype='_V', etype='_E', vtype='_U')
graph1.reverse()
print(graph1)
print(graph1.is_homogeneous)
print(graph1.is_block)
inp = torch.rand(3,3)
# print(g.is_unibipartite)
# g.ndata['_V']['in'] = inp
# print(g.edges())

graph = dgl.heterograph({('_U','_E','_V'):(torch.tensor([0,1,2]),torch.tensor([0,0,0]))})
print(graph.is_homogeneous)
print(graph.is_block)
graph.nodes['_U'].data['in'] = inp
# self.graph.nodes['_U'].data['in'] = f_in
graph.update_all(fn.copy_u('in', 'm'), fn.mean('m', 'out'))
