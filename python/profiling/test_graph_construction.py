
import torch, dgl
import time
from utils.utils import get_process_graph
# Get Graph
dg_graph, partition_map, num_classes = get_process_graph("ogbn-products", -1)
ind, indices, vals = dg_graph.adj_sparse('csr')
# Move everything
e1,e2 = dg_graph.adj_sparse('coo')
ind = ind.to('cuda:0')
indices = indices.to('cuda:0')
vals = vals.to('cuda:0')
e1 = e1.to('cuda:0')
e2 = e2.to('cuda:0')
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for i in range(10):
    start.record()
    t1 = time.time()
    g1 = dgl.heterograph({('U','E','V'):(e1,e2)})
    end.record()
    t2 = time.time()
    g2 = dgl.heterograph({('U','E','V'):('csr',(ind,indices,[]))})
    t3 = time.time()
    print("graph construction cost COO", t2-t1 )
    print("graph construction cost CSR", t3-t2 )
    g1.reverse()
    t4 = time.time()
    g2.reverse()
    t5 = time.time()
    print("COO reverse",t4-t3)
    print("CSR reverse",t5-t4)
# get graph
# t1 = time.time()
# h = []
# for i in range(4):
#     h.append(dgl.heterograph({('U','E','V'):('csr',(ind,indices,[]))}))
# t2 = time.time()
# print(t2-t1, "single construction")
# data = {}
#
# for i in range(4):
#     data[('U'+str(i),'E'+str(i),'V' + str(i))] = ('csr',(ind,indices,[]))
# h1 = dgl.heterograph(data)
# print(h1.num_nodes())
# t3 = time.time()
# print(t3-t2, "grouped construction")
# Check time for construction with COO and CSR
# With sparse matrix as well. try full graph
