import torch
import dgl
import dgl.function as fn
from dgl.heterograph import DGLHeteroGraph
@torch.jit.script
def func(g: DGLHeteroGraph):
    f = torch.rand(100,128).to(0)
    g.ndata['in'] = f
    g.update_all(fn.copy_u('in', 'm'), fn.sum('m', 'out'))
    print("Done")
            

if __name__ == "__main__":
    g = dgl.rand_graph(100,1000).to(0)
    func(g)
