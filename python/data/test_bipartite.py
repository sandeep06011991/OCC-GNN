
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
            continue
        from_dict[gpu_id] = gpu_id
        to_dict[gpu_id] = gpu_id
    data = []
    indptr_start = len(data)
    data.append(indptr)
    indptr_end = len(data)
    expand_indptr_start = len(data)
    data.append(expand_indptr)
    expand_indptr_end = len(data)
    indices_start = len(data)
    data.append(indices)
    indices_end = len(data)
    in_nodes_start = len(data)
    data.append(in_nodes)
    in_nodes_end = len(data)
    out_nodes_start = len(data)
    data.append(out_nodes)
    out_nodes_end = len(data)
    owned_out_nodes_start = len(data)
    data.append(owned_out_nodes)
    owned_own_nodes_end = len(data)
    self_ids_in_start = len(data)
    data.append(self_ids_in_start)
    self_ids_in_end = len(data)
    self_ids_out_start = len(data)
    data.append(self_ids_out)
    self_ids_out_end = lend(data)
    self.self_ids_in_start = metadatalist[12]
    self.self_ids_in_end = metadatalist[13]
    self.self_ids_out_start = metadatalist[14]
    self.self_ids_out_end = metadatalist[15]
    self.num_nodes_v = self.out_nodes_end - self.out_nodes_start
    self.from_ids_start = {}
    self.from_ids_end = {}
    self.to_ids_start = {}
    self.to_ids_end = {}
    for i in range(4):
        self.from_ids_start[i] = metadatalist[16 + i]
        self.from_ids_end[i] = metadatalist[20 + i]
        self.to_ids_start[i] = metadatalist[24 + i]
        self.to_ids_end[i] = metadatalist[28 + i]
    self.data = data
    # returns a list of bipartite grapjs
    # dgl_graph = heterograph
    # Partition_map for all nodes
    # No reordering
    # bp_graphs = []
    # (src,dest) = dgl_graph.edges()
    # for gpu_id in range(4):
    #     partition_map[dest]
    # Returns bipartite map broken in accord



if __name__ == "__main__":
    unit_test()

if __name__ == "__main__":
    print("test outline")
