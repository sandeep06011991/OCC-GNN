import dgl
import torch

dtype = torch.int32
a = torch.tensor([0,3,4,4,4,4]).to(dtype = dtype)
b = torch.tensor([1,2,3,4]).to(dtype = dtype)
c = torch.tensor([0,1,2,3],dtype = dtype)

g = dgl.graph(('csr',(a,b,c)), num_nodes = 5)
g = g.formats('csr')

#print(g.adj_sparse('coo'))
#print(g.adj_sparse('csc'))
