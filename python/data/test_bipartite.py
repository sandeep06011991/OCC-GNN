
from data.cpu_compr_bipartite import *

# Hardcode bipartite graphs for
#  [0 1 2 3] layer l+1
# [0,5,1,6,2,7,1,8] layer l
# everything is connected to everything
def get_bipartite_graph(gpu_id):
    indptr = [0,2,4,6,8]
    indices = [0,1,0,1,0,1,0,1]
    expand_indptr = [0,0,1,1,2,2,3,3]
    num_in_nodes = 2
    num_out_nodes = 4
    out_nodes = [0,1,2,3]
    owned_out_nodes = [gpu_id]
    in_nodes = [0,1]
    from_dict = {}
    to_dict = {}
    self_ids_in = [0]
    self_ids_out = [gpu_id]
    in_degrees = [2]
    for i in range(4):
        if i== gpu_id:
            from_dict[i] = []
            to_dict[i] = []
        else:
            from_dict[i] = [i]
            to_dict[i] = [i]

    data = [indptr, expand_indptr, indices, in_nodes, out_nodes, owned_out_nodes,
            self_ids_in, self_ids_out, in_degrees, from_dict[0], from_dict[1], from_dict[2],
            from_dict[3], to_dict[0], to_dict[1], to_dict[2], to_dict[3]]
    metadatalist = []
    newData = []
    for dataset in data:
        metadatalist.append(len(newData))
        newData.extend(dataset)
        metadatalist.append(len(newData))
    graph =  dgl.heterograph({('_V', '_E', '_U'): (expand_indptr, indices)},
                                 {'_U': num_in_nodes, '_V': num_out_nodes})
    graph = graph.reverse()
    graph = graph.formats('csc')
    return Bipartite.deserialize((torch.tensor(metadatalist+newData), graph, gpu_id))

def unit_test():
    for i in range(4):
        get_bipartite_graph(i)

if __name__ == "__main__":
    unit_test()
