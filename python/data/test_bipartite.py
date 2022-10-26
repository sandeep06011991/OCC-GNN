
from data.bipartite import *
from data.part_sample import *
from data.serialize import *
import torch

# Hardcode bipartite graphs for
#  [0 1 2 3] layer l+1
# [0,5,1,6,2,7,1,8] layer l
# everything is connected to everything
class cobject:
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.indptr = torch.tensor([0, 2, 4, 6, 8], device = self.gpu_id, dtype = torch.long)
        # Fix me:
        # Self nodes are now handled inpependent of the data graph
        # Therefore must not appear in indices and indptr
        self.indices = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device = self.gpu_id, dtype = torch.long)
        self.expand_indptr = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device = self.gpu_id, dtype = torch.long)
        self.num_in_nodes = 2
        self.indegree = torch.tensor([3], device = self.gpu_id, dtype = torch.long)
        self.num_out_nodes = 4
        self.gpu_id = gpu_id
        self.out_nodes = torch.tensor([0, 1, 2, 3], device = self.gpu_id, dtype = torch.long)
        self.owned_out_nodes = torch.tensor([gpu_id], device = self.gpu_id, dtype = torch.long)
        self.in_nodes = torch.tensor([0, 1], device = self.gpu_id, dtype = torch.long)
        self.from_ids = {}
        self.to_ids = {}
        self.self_ids_in = torch.tensor([0], device = self.gpu_id, dtype = torch.long)
        self.self_ids_out = torch.tensor([gpu_id], device = self.gpu_id, dtype = torch.long)
        self.in_degrees = torch.tensor([2], device = self.gpu_id, dtype = torch.long)
        for i in range(4):
            if i == gpu_id:
                self.from_ids[i] = torch.tensor([], device = self.gpu_id, dtype = torch.long)
                self.to_ids[i] = torch.tensor([], device = self.gpu_id, dtype = torch.long)
            else:
                self.from_ids[i] = torch.tensor([gpu_id], device = self.gpu_id, dtype = torch.long)
                self.to_ids  [i] = torch.tensor([i], device = self.gpu_id, dtype = torch.long)


def get_local_bipartite_graph(gpu_id):
    bp = Bipartite()
    bp.construct_from_cobject(cobject(gpu_id))
    return bp

def get_dummy_bipartite_graph():
    bp = Bipartite()
    cobj = cobject()
    bp.construct_from_cobject(cobj)
    return bp

def serialization_test():
    bp = get_dummy_bipartite_graph()
    tensor = serialize_to_tensor(bp)
    tensor = tensor.to(0)
    device = torch.device(0)
    bp_recon = Bipartite()
    construct_from_tensor_on_gpu(tensor, device, bp_recon)
    print("Test Success !!!")

def serializtion_test_gpu_local_sample():
    sample = Gpu_Local_Sample()
    sample.layers = [get_dummy_bipartite_graph() for i in range(3)]
    tensor = serialize_to_tensor(sample)
    tensor = tensor.to(0)
    tensor = tensor.long()
    device = torch.device(0)
    construct_from_tensor_on_gpu(tensor, device, sample)
    print(sample.layers[0].owned_out_nodes)
    print("Bipartite reconstruction also success !")
    pass

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
