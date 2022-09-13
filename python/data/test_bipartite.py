
from data.cpu_compr_bipartite import *

# Hardcode bipartite graphs for
#  [0 1 2 3] layer l+1
# [0,5,1,6,2,7,1,8] layer l
# everything is connected to everything
class cobject:
    indptr = [0, 2, 4, 6, 8]
    # Fix me:
    # Self nodes are now handled inpependent of the data graph
    # Therefore must not appear in indices and indptr
    indices = [0, 1, 0, 1, 0, 1, 0, 1]
    expand_indptr = [0, 0, 1, 1, 2, 2, 3, 3]
    num_in_nodes = 2
    in_degrees = 3
    num_out_nodes = 4
    out_nodes = [0, 1, 2, 3]
    owned_out_nodes = [gpu_id]
    in_nodes = [0, 1]
    from_dict = {}
    to_dict = {}
    self_ids_in = [0]
    self_ids_out = [gpu_id]
    in_degrees = [2]
    for i in range(4):
        if i == gpu_id:
            from_dict[i] = []
            to_dict[i] = []
        else:
            from_dict[i] = [gpu_id]
            to_dict[i] = [i]


def get_bipartite_graph(gpu_id):
    # Test Bipartite and CSR and CSC graph construction
    Bipartite(cobject)
    # Check flow of serializtion and deserializaiton. Looks Goodself.
    Gpu_Local_Sample()

def unit_test():
    for i in range(4):
        get_bipartite_graph(i)

if __name__ == "__main__":
    unit_test()
