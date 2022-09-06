import dgl
import torch
import dgl.function as fn
'''
Edges | Compute | Move
10 0.00043605333069960273 8.221866687138874e-05
100 0.00043198399742444354 8.190933366616568e-05
1000 0.0004331840028365453 8.205866689483324e-05
10000 0.00043577066560586294 8.374933401743571e-05
100000 0.001411093294620514 4.9082667877276746e-05
'''

# run this code with two different environments
# and got the exact same result
def run_experiment(n, runs):
    #graph = dgl.rand_graph(n, n * 100)
    #graph = graph.to(0)
    torch.cuda.set_device(0)
    print("DGL Version",dgl.__version__)
    e1 = torch.cuda.Event(enable_timing = True)
    e2 = torch.cuda.Event(enable_timing = True)
    e3 = torch.cuda.Event(enable_timing = True)
    #graph.ndata['in'] = torch.rand((n,128),device = 0)
    edges = [10,100,1000,10000,100000]
    print("Edges | Compute | Move")
    for num_edges in edges:
        u = [0 for i in range(num_edges)]
        v = [i for i in range(num_edges)]
        g = dgl.bipartite((v,u)).to(0)
        g.srcdata['in'] = torch.ones((num_edges,16)).to(0)
        graph = g
        compute_time = []
        move_time = []
        for j in range(runs):
            e1.record()
            with graph.local_scope():
                graph.update_all(fn.copy_src(src = 'in', out = 'm'),\
                    fn.sum(msg = 'm', out = 'h1'))
                e2.record()
                out = graph.dstdata['h1'].to('cpu')
            e3.record()
            e3.synchronize()
            
            compute_time.append(e1.elapsed_time(e2)/1000)
            move_time.append(e2.elapsed_time(e3)/1000)
        print(num_edges, sum(compute_time[2:])/(runs-2), sum(move_time[2:])/(runs-2))

if __name__ == "__main__":
    run_experiment(100,8)
