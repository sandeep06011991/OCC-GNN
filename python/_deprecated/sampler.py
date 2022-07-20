# Design of neighbour sampler
# With novel data_structureself.
import time
import dgl
import numpy as np
import scipy
from dgl._deprecate.graph import DGLGraph as DGLGraph
from dgl.contrib.sampling import NeighborSampler as NeighborSampler
import torch
from dgl.nn import GraphConv

def get_graph():
    DATA_DIR = "/home/spolisetty/data"
    graphname = "ogbn-products"
    indptr = np.fromfile("{}/{}/indptr.bin".format(DATA_DIR,graphname),dtype = np.intc)
    indices = np.fromfile("{}/{}/indices.bin".format(DATA_DIR,graphname),dtype = np.intc)
    num_nodes = indptr.shape[0] - 1
    num_edges = indices.shape[0]
    fsize = 600
    features = torch.rand(num_nodes,fsize)
    sp = scipy.sparse.csr_matrix((np.ones(indices.shape),indices,indptr),
    shape = (num_nodes,num_nodes))
    dg_graph = DGLGraph(sp)
    dg_graph.readonly()
    p_map = np.fromfile("{}/{}/partition_map.bin".format(DATA_DIR,graphname),dtype = np.intc)
    edges = dg_graph.edges()
    partition_map = torch.from_numpy(p_map)
    return dg_graph, partition_map, features, num_nodes, num_edges

dg_graph,partition_map, features, num_nodes, num_edges = get_graph()
fsize = features.shape[1]
# Create distributed torch tensor
class DistributedTensorMap():
    def __init__(self,global_to_gpu_id):
        # assert(tensor.shape[0] == global_to_gpu_id.shape[0])
        self.global_to_gpu_id = global_to_gpu_id
        self.local_to_global_id = []
        self.local_sizes = []
        for i in range(4):
            self.local_to_global_id.append(torch.where(self.global_to_gpu_id == i)[0])
            # print(local_to_global_id[i])
            self.local_sizes.append(self.local_to_global_id[i].shape[0])
        self.global_to_local_id = torch.zeros(global_to_gpu_id.shape,dtype = torch.int)
        for i in range(4):
            self.global_to_local_id.index_put_(indices = [self.local_to_global_id[i]],
                values = torch.arange(self.local_sizes[i],dtype = torch.int))

# dist_map1 = DistributedTensorMap(partition_map)
hops = 2
dataloader = (NeighborSampler(
        dg_graph, 4096, expand_factor = 10, num_hops = hops,
        shuffle = True))
total_1 = time.time()
reordering_time = []
forward_pass_time = []
intermediate_data = []
cost_of_data_movement = []
pa_graph = []
epoch_no = 0
print("total batches",num_nodes/4096)
# To avoud counting Dont count first epoch mem alloc
for i in range(4):
    a = torch.ones(1000 * 1000 * 1000,device=i)
    del a

for nf in dataloader:
    print("epoch no",epoch_no)
    epoch_no = epoch_no + 1
    if epoch_no > 100:
        break
    # s1 = time.time()
    # nf = next(dataloader)
    # e1 = time.time()
    # print("sample time",e1-s1)
    num_layers = nf.num_layers
    layer_offsets = [nf.layer_size(i) for i in range(num_layers)]
    edge_offsets = [nf.block_size(i) for i in range(nf.num_blocks)]
    s = 0
    for i in range(num_layers):
        t = s
        s = s + layer_offsets[i]
        layer_offsets[i] = t
    s = 0
    for i in range(nf.num_blocks):
        t = s
        s = s + edge_offsets[i]
        edge_offsets[i] = t

    # print(edge_offsets)
        # { for value in variable}
    # print(layer_offsets)
    dist_maps = []
    for i in range(hops + 1):
        dist_maps.append(DistributedTensorMap(partition_map[nf.layer_parent_nid(i)]))
    local_graphs = []
    merge_indices = []
    for i in range(hops):
        local_graphs.append([[None for i in range(4)] for i in range(4)])
        merge_indices.append([[None for i in range(4)] for i in range(4)])

    #
    s1_time = time.time()
    features[nf.layer_parent_nid(0)].to('cuda:0')
    e1_time = time.time()
    cost_of_data_movement.append(e1_time - s1_time)
    # dist_map1 = DistributedTensorMap(partition_map[a.layer_parent_nid(0)])
    # dist_map2 = DistributedTensorMap(partition_map[a.layer_parent_nid(1)])
    # dist_map3 = DistributedTensorMap(partition_map[a.layer_parent_nid(2)])
    #
    # local_graphs1 = [[None for i in range(4)] for i in range(4)]
    # merge_indices1 = [[None for i in range(4)] for i in range(4)]
    # print("Beggining construction of local graphs")
    # with torch.autograd.profiler.profile() as prof:
    if(True):
        start_time = time.time()
        for i in range(hops):
            # print("Working on hops !!")
            current_layer_id = i
            next_layer_id = i+1
            src_parent_id = nf.layer_parent_nid(current_layer_id)
            dest_parent_id = nf.layer_parent_nid(next_layer_id)
            src, dest, edge_id = nf.block_edges(current_layer_id)
            block_id = current_layer_id
            # print(src)
            # print(dest)
            # print(dist_maps[current_layer_id].global_to_gpu_id.shape)
            # print(dist_maps[next_layer_id].global_to_gpu_id.shape)
            edge_map = (dist_maps[current_layer_id].global_to_gpu_id[src - layer_offsets[current_layer_id]]), \
                dist_maps[next_layer_id].global_to_gpu_id[dest - layer_offsets[next_layer_id]]
            dist_map1 = dist_maps[current_layer_id]
            dist_map2 = dist_maps[next_layer_id]
            for src_gpu in range(4):
                for dest_gpu in range(4):
                    if src_gpu == dest_gpu:
                        local_edges = edge_id[torch.where((edge_map[0] == src_gpu) & (edge_map[1]==dest_gpu))] - edge_offsets[block_id]
                        src_local_edges = dist_map1.global_to_local_id[src[local_edges]-layer_offsets[current_layer_id]]
                        dest_local_edges = dist_map2.global_to_local_id[dest[local_edges]-layer_offsets[next_layer_id]]
                        num_src_nodes = dist_map1.local_sizes[src_gpu]
                        num_dest_nodes = dist_map2.local_sizes[dest_gpu]
                        # print("local",num_dest_nodes)
                        block = dgl.create_block((src_local_edges,dest_local_edges),num_src_nodes = num_src_nodes, \
                                                num_dst_nodes = num_dest_nodes).to(src_gpu)
                        local_graphs[current_layer_id][src_gpu][dest_gpu] = block
                    else:
                        non_local_edges = edge_id[torch.where((edge_map[0] == src_gpu) & (edge_map[1] == dest_gpu))] - edge_offsets[block_id]
                        # print("edges",non_local_edges.shape[0],src_gpu)
                        src_local_edges = dist_map1.global_to_local_id[src[non_local_edges]-layer_offsets[current_layer_id]]
                        remote_dest_nodes = dest[non_local_edges]-layer_offsets[next_layer_id]
                        no_dup,mapping = remote_dest_nodes.unique(return_inverse = True)
                        merge_indices[current_layer_id][src_gpu][dest_gpu] = dist_map2.global_to_local_id[no_dup].long().to(dest_gpu)
                        num_dest_nodes = no_dup.shape[0]
                        num_src_nodes = dist_map1.local_sizes[src_gpu]
                        dest_local_edges = mapping.type(torch.int32)
                        # print("non-local dest",num_dest_nodes)
                        # print("non-local src",num_src_nodes)
                        block = dgl.create_block((src_local_edges,dest_local_edges),
                                num_src_nodes = num_src_nodes, num_dst_nodes = num_dest_nodes).to(src_gpu)
                        local_graphs[current_layer_id][src_gpu][dest_gpu] = block
        end_time = time.time()
        reordering_time.append(end_time-start_time)

    # print("local ordering ",end_time - start_time)
    # print("local graphs all created")
    # print("Begin forward pass !!")
    global_f = features[nf.layer_parent_nid(0)]
    local_features = []
    conv1 = []
    conv2 = []
    conv3 = []
    for i in range(4):
        local_features.append(global_f[dist_maps[0].local_to_global_id[i]].to(i))
        conv1.append(GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True).to(i))
        conv2.append(GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True).to(i))
        conv3.append(GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True).to(i))
    # print("local slices done")


    # features[a.layer_parent_nid(0)[:100]].to(0)
    # s2 = time.time()
    # f2 = features[a.layer_parent_nid(0)].to(0)
    # e2 = time.time()
    # print("total time feature transfer",e2 - s2)
    def forward_pass(distributed_input, per_gpu_conv_list, local_graphs, merge_indices,timer):
        # print("Working on fpass")
        forward_pass_layer_1 = [[None for i in range(4)] for i in range(4)]
        for src in range(4):
            for dest in range(4):
                forward_pass_layer_1[src][dest] =conv1[src](local_graphs[src][dest],distributed_input[src])
        # print("Forward pass done !")
        # print("shuffle and merge !!")
        s1 = time.time()
        for src in range(4):
            for dest in range(4):
                if(src != dest):
                    temp = forward_pass_layer_1[src][dest].to(dest)
                    final = forward_pass_layer_1[dest][dest]
                    # print(merge_indices1)
                    final[merge_indices[src][dest]] += temp
        s2 = time.time()
        timer.append(s2-s1)
        out = []
        for i in range(4):
            out.append(forward_pass_layer_1[i][i])
        return out
    t1 = time.time()
    out1 = forward_pass(local_features,conv1, local_graphs[0],merge_indices[0],intermediate_data)
    out2 = forward_pass(out1,conv2, local_graphs[1],merge_indices[1],intermediate_data)
    if hops==3:
        out3 = forward_pass(out2,conv3,local_graphs[2],merge_indices[2],intermediate_data)
    # for i in range(out2):
    #     print(i.device)
    #     i.sum().backward()
    # print("shuffle and merge done")
    t2 = time.time()
    forward_pass_time.append(t2-t1)
    # print("Total time for forward pass",t2-t1)
    # print("hello world")
print("total training time",time.time()-total_1)
print("Total reordering_time",sum(reordering_time))
print("total forward tiem",sum(forward_pass_time))
print("total intermediate time",sum(intermediate_data))
print("all data movement ",sum(cost_of_data_movement ))
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# assert(False)
# except StopIteration:
#     pass
# except e:
#     print(e)
