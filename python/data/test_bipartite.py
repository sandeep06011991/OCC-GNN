
from cpu_compr_bipartite import *

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
    owned_out_nodes = [i]
    in_nodes = [0,1]
    from_dict = {}
    to_dict = {}
    self_ids_in = [0]
    self_ids_out = [i]
    for i in range(4):
        if i== gpu_id:
            from_dict[gpu_id] = []
            to_dict[gpu_id] = []
        from_dict[gpu_id] = [gpu_id]
        to_dict[gpu_id] = [gpu_id]

    data = [indptr, expand_indptr, indices, in_nodes, out_nodes, owned_out_nodes,
            self_ids_in, self_ids_out, from_dict[0], from_dict[1], from_dict[2],
            from_dict[3], to_dict[0], to_dict[1], to_dict[2], to_dict[3]]
    metadatalist = []
    newData = []
    for dataset in data:
        metadatalist.append(len(dataset))
        newData.extend(dataset)
        metadatalist.append(len(dataset))
    return Bipartite.deserialize(newData, metadatalist)

if __name__ == "__main__":
    unit_test()

if __name__ == "__main__":
    print("test outline")
