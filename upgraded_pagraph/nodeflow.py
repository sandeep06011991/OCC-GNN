import dgl
from dgl._deprecate.graph import DGLGraph
#nodeflow is a deprecated object whos poperties are not clear

g = DGLGraph(([0,1,1,2,3,4],[1,2,3,4,5,6]))
g.readonly()
sampler = dgl.contrib.sampling.NeighborSampler(g, 1,1)

ss = iter(sampler)
nf = next(ss)
#nf.blocks[0]
#print(nf.layers[0].number_of_nodes())
#print(nf.blocks[0].number_of_edges())
