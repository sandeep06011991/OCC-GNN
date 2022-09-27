import dgl
 
#nodeflow is a deprecated object whos poperties are not clear

g = dgl.DGLGraph(([0,1,1,2,3,4],[1,2,3,4,5,6]))
g.readonly()
sampler = dgl.contrib.sampling.NeighborSampler(g, 1,1)

ss = iter(sampler)
nf = next(ss)
#print(nf.layers[0].num_of_nodes())
