import time
import dgl
import numpy as np
import scipy
from dgl import DGLGraph
# from dgl.contrib.sampling import NeighborSampler as NeighborSampler
import torch
import torch.nn as nn
# from dgl.nn import GraphConv
# from dgl.nn import SageConv as GraphConv
from dgl.nn.pytorch import SAGEConv as GraphConv
from utils import get_dgl_graph
import argparse

# dg_graph,partition_map, features, num_nodes, num_edges = get_graph()
# fsize = features.shape[1]
# Create distributed torch tensor
class DistributedTensorMap():
    # Datastructure to store mapping of a global tensor split across
    # available GPUs
    def __init__(self, global_to_gpu_id, tensor = None):
        # assert(tensor.shape[0] == global_to_gpu_id.shape[0])
        self.global_to_gpu_id = global_to_gpu_id
        self.local_to_global_id = []
        self.local_sizes = []
        if tensor != None:
            assert(tensor.device == torch.device("cpu"))
        for i in range(4):
            self.local_to_global_id.append(torch.where(self.global_to_gpu_id == i)[0])
            # print(local_to_global_id[i])
            self.local_sizes.append(self.local_to_global_id[i].shape[0])
        self.global_to_local_id = torch.zeros(global_to_gpu_id.shape,dtype = torch.int)
        self.local_tensors = []
        for i in range(4):
            self.global_to_local_id.index_put_(indices = [self.local_to_global_id[i]],
                values = torch.arange(self.local_sizes[i],dtype = torch.int))
            if tensor != None:
                self.local_tensors.append(tensor[self.local_to_global_id[i]].to(torch.device(i)))

class DistGraphConv(nn.Module):
    def __init__(self,in_d, out):
        super(DistGraphConv, self).__init__()
        self.layers = [GraphConv(in_d, out, \
            norm="none", aggregator_type = "mean").to(i) for i in range(4)]

    def forward(self, distributed_input,  \
            local_graphs, merge_indices):
        forward_pass_layer_1 = [[None for i in range(4)] for i in range(4)]
        for src in range(4):
            for dest in range(4):
                forward_pass_layer_1[src][dest] = \
                    self.layers[src](local_graphs[src][dest],distributed_input[src])
        # print("Forward pass done !")
        # print("shuffle and merge !!")
        for src in range(4):
            for dest in range(4):
                if(src != dest):
                    temp = forward_pass_layer_1[src][dest].to(dest)
                    final = forward_pass_layer_1[dest][dest]
                    # print(merge_indices1)
                    final[merge_indices[src][dest]] += temp
        out = []
        for i in range(4):
            out.append(forward_pass_layer_1[i][i])
        return sliced_tensor

class Model(nn.Module):
    def __init__(self, no_layers, fsize,  dist_feature, p_map):
        super(Model, self).__init__()
        self.no_layers = no_layers
        self.layers = [DistGraphConv(fsize, fsize) for i in range(no_layers)]
        self.dist_feature = dist_feature
        self.p_map = p_map

    def forward(self,in_nodes, out_nodes, blocks):
        local_graphs,  merge_indices = self.create_local_graphs(in_nodes,
                        out_nodes, blocks)
        in_map = self.layers[0](self.dist_feature.local_tensors,
                    local_graph[0], merge_indices[0])
        no_layers = len(blocks)
        for i in range(1,no_layers):
            in_map = self.layers[i](in_map, local_graphs[i], \
                        merge_indices[i])

    def create_local_graphs(self, in_nodes, out_nodes,blocks):
        num_layers = len(blocks)
        partition_map = self.p_map
        dist_feature = self.dist_feature
        local_blocks = []
        merge_indices = []
        dist_maps = [dist_feature]
        for i in range(1,num_layers):
            dist_maps.append(DistributedTensorMap(partition_map[blocks[i].ndata["_ID"]["_U"]]))
        dist_maps.append(DistributedTensorMap(partition_map[out_nodes]))
        for i in range(num_layers):
            print("working on layer",i)
            block = blocks[i]
            block_in = blocks[i].ndata["_ID"]["_U"]
            if i != num_layers - 1:
                block_out = blocks[i+1].ndata["_ID"]["_U"]
            else:
                block_out = out_nodes
            src_nds, dest_nodes = block.edges()
            dist_map_in = dist_maps[i]
            dist_map_out = dist_maps[i+1]
            if i==0:
                edge_src_partition = dist_map_in.global_to_gpu_id[block_in[src_nds]]
            else:
                edge_src_partition = dist_map_in.global_to_gpu_id[src_nds]
            edge_dest_partition = dist_map_out.global_to_gpu_id[dest_nodes]
            edge_map = edge_src_partition, edge_dest_partition
            local_graph_per_layer = [[i for i in range(4)] for i in range(4)]
            merge_indices_per_layer = [[i for i in range(4)] for i in range(4)]
            for src_gpu in range(4):
                for dest_gpu in range(4):
                    if src_gpu == dest_gpu:
                        local_edges = torch.where((edge_map[0] == src_gpu) & (edge_map[1]==dest_gpu))
                        src_local_edges = dist_map_in.global_to_local_id[src_nds[local_edges]]
                        dest_local_edges = dist_map_out.global_to_local_id[dest_nds[local_edges]]
                        num_src_nodes = dist_map1.local_sizes[src_gpu]
                        num_dest_nodes = dist_map2.local_sizes[dest_gpu]
                        # print("local",num_dest_nodes)
                        block = dgl.create_block((src_local_edges,dest_local_edges),num_src_nodes = num_src_nodes, \
                                                num_dst_nodes = num_dest_nodes).to(src_gpu)
                        local_graph_per_layer[src_gpu][dest_gpu] = block
                    else:
                        non_local_edges = torch.where((edge_map[0] == src_gpu) & (edge_map[1] == dest_gpu))
                        # print("edges",non_local_edges.shape[0],src_gpu)
                        src_local_edges = dist_map1.global_to_local_id[src[non_local_edges]]
                        remote_dest_nodes = dest[non_local_edges]
                        no_dup,mapping = remote_dest_nodes.unique(return_inverse = True)
                        merge_indices[src_gpu][dest_gpu] = dist_map2.global_to_local_id[no_dup].long().to(dest_gpu)
                        num_dest_nodes = no_dup.shape[0]
                        num_src_nodes = dist_map1.local_sizes[src_gpu]
                        dest_local_edges = mapping.type(torch.int32)
                        # print("non-local dest",num_dest_nodes)
                        # print("non-local src",num_src_nodes)
                        block = dgl.create_block((src_local_edges,dest_local_edges),
                                num_src_nodes = num_src_nodes, num_dst_nodes = num_dest_nodes).to(src_gpu)
                        local_graph_per_layer[src_gpu][dest_gpu] = block
            local_blocks.append(local_graph_per_layer)
            merge_indices.append(merge_indices_per_layer)
        assert(len(local_blocks) == len(blocks))
        return local_blocks, merge_indices


# dist_map1 = DistributedTensorMap(partition_map)
def train(dg_graph, p_map, args):
    hops = args.num_layers
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [10 for i in range(hops)])
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        torch.arange(0,dg_graph.num_nodes()),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    # fixme:
    print("Note: Current dgl version performs async sampling + slicing")
    print("Pass a graph without features to DGL to bypass this step ")
    input_distributed = DistributedTensorMap(p_map, dg_graph.ndata["features"])
    model = Model(hops, args.fsize, input_distributed, p_map)
    epoch_no = 0
    print("total batches",dg_graph.num_nodes()/args.batch_size)
    # To avoid counting Dont count first epoch mem alloc
    # for i in range(4):
    #     a = torch.ones(1000 * 1000 * 1000,device=i)
    #     del a

    for epoch_no,(in_nodes,out_nodes,blocks) in enumerate(dataloader):
        print("epoch no",epoch_no)
        # assert(False)
        epoch_no = epoch_no + 1
        if epoch_no > 100:
            break
        # assert(False)
        model(in_nodes, out_nodes, blocks)
        print("Forward pass successful !!")
        assert(False)
        # s1 = time.time()
        # nf = next(dataloader)
        # e1 = time.time()
        # # print("sample time",e1-s1)
        # num_layers = nf.num_layers
        # layer_offsets = [nf.layer_size(i) for i in range(num_layers)]
        # edge_offsets = [nf.block_size(i) for i in range(nf.num_blocks)]
        # s = 0
        # for i in range(num_layers):
        #     t = s
        #     s = s + layer_offsets[i]
        #     layer_offsets[i] = t
        # s = 0
        # for i in range(nf.num_blocks):
        #     t = s
        #     s = s + edge_offsets[i]
        #     edge_offsets[i] = t

        # print(edge_offsets)
            # { for value in variable}
        # print(layer_offsets)
        # dist_maps = []
        # for i in range(hops + 1):
        #     dist_maps.append(DistributedTensorMap(partition_map[nf.layer_parent_nid(i)]))
        # local_graphs = []
        # merge_indices = []
        # for i in range(hops):
        #     local_graphs.append([[None for i in range(4)] for i in range(4)])
        #     merge_indices.append([[None for i in range(4)] for i in range(4)])

        # #
        # s1_time = time.time()
        # features[nf.layer_parent_nid(0)].to('cuda:0')
        # e1_time = time.time()
        # cost_of_data_movement.append(e1_time - s1_time)
        # dist_map1 = DistributedTensorMap(partition_map[a.layer_parent_nid(0)])
        # dist_map2 = DistributedTensorMap(partition_map[a.layer_parent_nid(1)])
        # dist_map3 = DistributedTensorMap(partition_map[a.layer_parent_nid(2)])
        #
        # local_graphs1 = [[None for i in range(4)] for i in range(4)]
        # merge_indices1 = [[None for i in range(4)] for i in range(4)]
        # print("Beggining construction of local graphs")
        # with torch.autograd.profiler.profile() as prof:
        # if(True):
        #     start_time = time.time()
        #     for i in range(hops):
        #         # print("Working on hops !!")
        #         current_layer_id = i
        #         next_layer_id = i+1
        #         src_parent_id = nf.layer_parent_nid(current_layer_id)
        #         dest_parent_id = nf.layer_parent_nid(next_layer_id)
        #         src, dest, edge_id = nf.block_edges(current_layer_id)
        #         block_id = current_layer_id
        #         edge_map = (dist_maps[current_layer_id].global_to_gpu_id[src - layer_offsets[current_layer_id]]), \
        #             dist_maps[next_layer_id].global_to_gpu_id[dest - layer_offsets[next_layer_id]]
        #         dist_map1 = dist_maps[current_layer_id]
        #         dist_map2 = dist_maps[next_layer_id]
        #         for src_gpu in range(4):
        #             for dest_gpu in range(4):
        #                 if src_gpu == dest_gpu:
        #                     local_edges = edge_id[torch.where((edge_map[0] == src_gpu) & (edge_map[1]==dest_gpu))] - edge_offsets[block_id]
        #                     src_local_edges = dist_map1.global_to_local_id[src[local_edges]-layer_offsets[current_layer_id]]
        #                     dest_local_edges = dist_map2.global_to_local_id[dest[local_edges]-layer_offsets[next_layer_id]]
        #                     num_src_nodes = dist_map1.local_sizes[src_gpu]
        #                     num_dest_nodes = dist_map2.local_sizes[dest_gpu]
        #                     # print("local",num_dest_nodes)
        #                     block = dgl.create_block((src_local_edges,dest_local_edges),num_src_nodes = num_src_nodes, \
        #                                             num_dst_nodes = num_dest_nodes).to(src_gpu)
        #                     local_graphs[current_layer_id][src_gpu][dest_gpu] = block
        #                 else:
        #                     non_local_edges = edge_id[torch.where((edge_map[0] == src_gpu) & (edge_map[1] == dest_gpu))] - edge_offsets[block_id]
        #                     # print("edges",non_local_edges.shape[0],src_gpu)
        #                     src_local_edges = dist_map1.global_to_local_id[src[non_local_edges]-layer_offsets[current_layer_id]]
        #                     remote_dest_nodes = dest[non_local_edges]-layer_offsets[next_layer_id]
        #                     no_dup,mapping = remote_dest_nodes.unique(return_inverse = True)
        #                     merge_indices[current_layer_id][src_gpu][dest_gpu] = dist_map2.global_to_local_id[no_dup].long().to(dest_gpu)
        #                     num_dest_nodes = no_dup.shape[0]
        #                     num_src_nodes = dist_map1.local_sizes[src_gpu]
        #                     dest_local_edges = mapping.type(torch.int32)
        #                     # print("non-local dest",num_dest_nodes)
        #                     # print("non-local src",num_src_nodes)
        #                     block = dgl.create_block((src_local_edges,dest_local_edges),
        #                             num_src_nodes = num_src_nodes, num_dst_nodes = num_dest_nodes).to(src_gpu)
        #                     local_graphs[current_layer_id][src_gpu][dest_gpu] = block
        #     end_time = time.time()
        #     reordering_time.append(end_time-start_time)
        #
        # # print("local ordering ",end_time - start_time)
        # # print("local graphs all created")
        # # print("Begin forward pass !!")
        # global_f = features[nf.layer_parent_nid(0)]
        # local_features = []
        # conv1 = []
        # conv2 = []
        # conv3 = []
        # for i in range(4):
        #     local_features.append(global_f[dist_maps[0].local_to_global_id[i]].to(i))
        #     conv1.append(GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True).to(i))
        #     conv2.append(GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True).to(i))
        #     conv3.append(GraphConv(fsize,fsize, norm="none",allow_zero_in_degree = True).to(i))
        # print("local slices done")


        # features[a.layer_parent_nid(0)[:100]].to(0)
        # s2 = time.time()
        # f2 = features[a.layer_parent_nid(0)].to(0)
        # e2 = time.time()
        # print("total time feature transfer",e2 - s2)

    #     t1 = time.time()
    #     out1 = forward_pass(local_features,conv1, local_graphs[0],merge_indices[0],intermediate_data)
    #     out2 = forward_pass(out1,conv2, local_graphs[1],merge_indices[1],intermediate_data)
    #     if hops==3:
    #         out3 = forward_pass(out2,conv3,local_graphs[2],merge_indices[2],intermediate_data)
    #     # for i in range(out2):
    #     #     print(i.device)
    #     #     i.sum().backward()
    #     # print("shuffle and merge done")
    #     t2 = time.time()
    #     forward_pass_time.append(t2-t1)
    #     # print("Total time for forward pass",t2-t1)
    #     # print("hello world")
    # print("total training time",time.time()-total_1)
    # print("Total reordering_time",sum(reordering_time))
    # print("total forward tiem",sum(forward_pass_time))
    # print("total intermediate time",sum(intermediate_data))
    # print("all data movement ",sum(cost_of_data_movement ))
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# assert(False)
# except StopIteration:
#     pass
# except e:
#     print(e)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, default = "ogbn-arxiv")
    argparser.add_argument('--fsize',type = int,default = 1024)
    argparser.add_argument('--gpu', type=str,
                            default = 0,
                            # default='0,1,2,3',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=1024)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    # argparser.add_argument('--data-cpu', action='store_true', default = True,
    #                        help="By default the script puts all node features and labels "
    #                             "on GPU when using it to save time for data copy. This may "
    #                             "be undesired if they cannot fit in GPU memory at once. "
    #                             "This flag disables that.")
    args = argparser.parse_args()
    n_classes = 40
    g,p_map =  get_dgl_graph(args.graph)
    train(g,p_map,args)
