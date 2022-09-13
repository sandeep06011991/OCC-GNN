
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
import torch

# Hardcode bipartite graphs for
#  [0 1 2 3] layer l+1
# [0,5,1,6,2,7,1,8] layer l
# everything is connected to everything
class cobject:
    indptr = torch.tensor([0, 2, 4, 6, 8])
    # Fix me:
    # Self nodes are now handled inpependent of the data graph
    # Therefore must not appear in indices and indptr
    indices = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    expand_indptr = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    num_in_nodes = 2
    indegree = torch.tensor([3])
    num_out_nodes = 4
    gpu_id = 0
    out_nodes = torch.tensor([0, 1, 2, 3])
    owned_out_nodes = torch.tensor([gpu_id])
    in_nodes = torch.tensor([0, 1])
    from_ids = {}
    to_ids = {}
    self_ids_in = torch.tensor([0])
    self_ids_out = torch.tensor([gpu_id])
    in_degrees = torch.tensor([2])
    for i in range(4):
        if i == gpu_id:
            from_ids[i] = torch.tensor([])
            to_ids[i] = torch.tensor([])
        else:
            from_ids[i] = torch.tensor([gpu_id])
            to_ids  [i] = torch.tensor([i])

def get_dummy_bipartite_graph():
    bp = Bipartite()
    cobj = cobject()
    bp.construct_from_cobject(cobj)
    return bp

def serialization_test():
    bp = get_dummy_bipartite_graph()
    tensor = serialize_to_tensor(bp)
    tensor = tensor.to(0)
    device = 0
    bp_recon = Bipartite()
    construct_from_tensor_on_gpu(self, tensor, device, bp_recon)
    print("Test Success !!!")

# def get_dummy_gpu_local_graph():
#     pass
#
# def get_bipartite_graph(gpu_id):
#     # Test Bipartite and CSR and CSC graph construction
#     Bipartite(cobject)
#     # Check flow of serializtion and deserializaiton. Looks Goodself.
#     Gpu_Local_Sample()
#

if __name__ == "__main__":
    unit_test()
