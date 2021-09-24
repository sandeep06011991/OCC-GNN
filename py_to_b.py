import dgl
graphs = dgl.data.CoraFullDataset('../data')
# Test graph data
# mat = graphs[0].adj(scipy_fmt='csr')
# mat.sort_indices()
# print(mat.indptr[:5])
# fp = open('test.bin','wb')
# fp.write(mat.indptr[:5].tobytes())
# fp.close()

# data = graphs[0].ndata['feat'].numpy()
# print(data.shape)
# data[0][0] = .1
# data[0][1] = .2
# fp = open('test.bin','wb')
# print(data[0][:10])
# fp.write(data.tobytes())
# fp.close()

num_nodes = graphs[0].num_nodes()
num_edges = graphs[0].num_edges()
a = {"num_nodes":num_nodes,"num_edges":num_edges}
print(a)
f = open("meta.json", "w")
import json
f.write(json.dumps(a))
f.close()
