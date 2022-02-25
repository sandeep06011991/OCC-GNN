# import torch
#
# class DistTensorSlice():
#     # Takes in global ids and partition map and reorders them.
#     # Provides easy functions to find global to local id.
#     # Used only on l-1 layers of network.
#     # Dont need it on the lth layer as it uses the static cache and current ordering
#     def __init__(self, global_ids, partition_map, num_nodes):
#         # assert(tensor.shape[0] == global_to_gpu_id.shape[0])
#         self.global_id = global_id
#         self.global_to_gpu_id = partition_map
#         self.local_to_global_id = []
#         self.local_sizes = []
#         if tensor != None:
#             assert(tensor.device == torch.device("cpu"))
#         for i in range(4):
#             self.local_to_global_id.append(global_id[torch.where(self.global_to_gpu_id == i)[0]])
#             # print(local_to_global_id[i])
#             self.local_sizes.append(self.local_to_global_id[i].shape[0])
#         self.global_to_local_id = torch.zeros(num_nodes,dtype = torch.int)
#         self.local_tensors = []
#         for i in range(4):
#             self.global_to_local_id.index_put_(indices = [self.local_to_global_id[i]],
#                 values = torch.arange(self.local_sizes[i],dtype = torch.int))
