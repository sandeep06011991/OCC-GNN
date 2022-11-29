
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
        self.num_in_nodes_local = 2
        self.num_in_nodes_pulled = 0
        self.num_out_local = 1
        self.num_out_remote = 3

        self.indptr_L = torch.tensor([0,2], device =  gpu_id)
        self.indices_L = torch.tensor([0,1], device =  gpu_id)
        self.indptr_R = torch.tensor([0,2,4,6], device =  gpu_id)
        self.indices_R = torch.tensor([0,1,0,1,0,1], device =  gpu_id)
        self.out_degree_local = torch.tensor([8], device =  gpu_id)

        self.from_ids = {}
        self.push_to_ids = {}
        self.to_offsets = []
        self.pull_from_offsets = []
        self.to_offsets.append(0)
        self.pull_from_offsets.append(0)
        self.self_ids_offset = 1
        for i in range(4):
            if i == self.gpu_id:
                self.to_offsets.append(self.to_offsets[-1])
                self.pull_from_offsets.append(self.pull_from_offsets[-1])
            else:
                self.to_offsets.append(self.to_offsets[-1] + 1)
                self.pull_from_offsets.append(self.pull_from_offsets[-1])
            self.from_ids[i] = torch.tensor([0], device =  gpu_id)
            self.push_to_ids[i] = torch.tensor([], device =  gpu_id)

# Used to testing dist-GCN
def get_local_bipartite_graph(gpu_id):
    bp = Bipartite()
    bp.construct_from_cobject(cobject(gpu_id))
    bp.reconstruct_graph()
    return bp




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
