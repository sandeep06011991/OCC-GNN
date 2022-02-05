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
from dgl.nn.pytorch import SAGEConv
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
        self.layers = [SAGEConv(in_d, out, \
            norm=None, aggregator_type = "mean").to(i) for i in range(4)]
        self.forward_time = 0
        self.merge_time = 0
        self.data_movement = 0

    def forward(self, distributed_input,  \
            local_graphs, merge_indices):
        forward_pass_layer_1 = [[None for i in range(4)] for i in range(4)]
        t1 = time.time()
        for src in range(4):
            for dest in range(4):
                forward_pass_layer_1[src][dest] = \
                    self.layers[src](local_graphs[src][dest],distributed_input[src])
        t2 = time.time()
        self.forward_time = self.forward_time  + t2 - t1
        # print("Forward pass done !")
        # print("shuffle and merge !!")
        for src in range(4):
            for dest in range(4):
                if(src != dest):
                    t3 = time.time()
                    temp = forward_pass_layer_1[src][dest].to(dest)
                    t4 = time.time()
                    final = forward_pass_layer_1[dest][dest]
                    # print(merge_indices1)
                    final[merge_indices[src][dest]] += temp
                    t5 = time.time()
                    self.merge_time = self.merge_time + (t5 - t4)
                    self.data_movement = self.data_movement + (t4 - t3)
        local_sliced_tensor = []
        for i in range(4):
            local_sliced_tensor.append(forward_pass_layer_1[i][i])
        return local_sliced_tensor

    def clear_timers(self):
        self.forward_time = 0
        self.merge_time = 0
        self.data_movement = 0

class Model(nn.Module):
    def __init__(self, no_layers, fsize, hidden,  no_classes, dist_feature, p_map):
        super(Model, self).__init__()
        self.no_layers = no_layers
        self.layers = [DistGraphConv(fsize, hidden)]
        for i in range(1,no_layers-1):
            self.layers.append(DistGraphConv(hidden,hidden))
        self.layers.append(DistGraphConv(hidden,no_classes))
        self.dist_feature = dist_feature
        self.p_map = p_map
        self.graph_creation_time = 0

    def clear_timers(self):
        self.graph_creation_time = 0
        for i in self.layers:
            i.clear_timers()

    def get_graph_creation_time(self):
        return self.graph_creation_time

    def get_forward(self):
        s1 = 0
        s2 = 0
        s3 = 0
        for i in self.layers:
            s1 = s1 + i.forward_time
            s2 = s2 + i.merge_time
            s3 = s3 + i.data_movement
        return s1,s2,s3


    def forward(self,in_nodes, out_nodes, blocks):
        t1 = time.time()
        local_graphs,  merge_indices = self.create_local_graphs(in_nodes,
                        out_nodes, blocks)
        t2 = time.time()
        in_map = self.layers[0](self.dist_feature.local_tensors,
                    local_graphs[0], merge_indices[0])
        no_layers = len(blocks)
        self.graph_creation_time = self.graph_creation_time + t2 - t1
        return
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
            # print("working on layer",i)
            block = blocks[i]
            block_in = blocks[i].ndata["_ID"]["_U"]
            if i != num_layers - 1:
                block_out = blocks[i+1].ndata["_ID"]["_U"]
            else:
                block_out = out_nodes
            src_nds, dest_nds = block.edges()
            dist_map_in = dist_maps[i]
            dist_map_out = dist_maps[i+1]
            if i==0:
                # Ids in first layer use global graph ids
                edge_src_partition = dist_map_in.global_to_gpu_id[block_in[src_nds]]
            else:
                # Ids in other layers use local_ids
                edge_src_partition = dist_map_in.global_to_gpu_id[src_nds]
            edge_dest_partition = dist_map_out.global_to_gpu_id[dest_nds]
            edge_map = edge_src_partition, edge_dest_partition
            local_graph_per_layer = [[i for i in range(4)] for i in range(4)]
            merge_indices_per_layer = [[i for i in range(4)] for i in range(4)]
            for src_gpu in range(4):
                for dest_gpu in range(4):
                    if src_gpu == dest_gpu:
                        local_edges = torch.where((edge_map[0] == src_gpu) & (edge_map[1]==dest_gpu))
                        if i==0:
                            src_local_edges = dist_map_in.global_to_local_id[block_in[src_nds[local_edges]]]
                        else:
                            src_local_edges = dist_map_in.global_to_local_id[src_nds[local_edges]]
                        dest_local_edges = dist_map_out.global_to_local_id[dest_nds[local_edges]]
                        num_src_nodes = dist_map_in.local_sizes[src_gpu]
                        num_dest_nodes = dist_map_out.local_sizes[dest_gpu]
                        # print("local",num_dest_nodes)
                        block = dgl.create_block((src_local_edges,dest_local_edges),num_src_nodes = num_src_nodes, \
                                                num_dst_nodes = num_dest_nodes).to(src_gpu)
                        local_graph_per_layer[src_gpu][dest_gpu] = block
                    else:
                        non_local_edges = torch.where((edge_map[0] == src_gpu) & (edge_map[1] == dest_gpu))
                        # print("edges",non_local_edges.shape[0],src_gpu)
                        if i==0:
                            src_local_edges = dist_map_in.global_to_local_id[block_in[src_nds[non_local_edges]]]
                        else:
                            src_local_edges = dist_map_in.global_to_local_id[src_nds[non_local_edges]]
                        remote_dest_nodes = dest_nds[non_local_edges]
                        no_dup,mapping = remote_dest_nodes.unique(return_inverse = True)
                        merge_indices_per_layer[src_gpu][dest_gpu] = dist_map_out.global_to_local_id[no_dup].long().to(dest_gpu)
                        num_dest_nodes = no_dup.shape[0]
                        num_src_nodes = dist_map_in.local_sizes[src_gpu]
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
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
        # [10 for i in range(hops)])
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
    num_classes = 40
    model = Model(hops, args.fsize, args.num_hidden, num_classes, input_distributed, p_map)
    epoch_no = 0
    print("total batches",dg_graph.num_nodes()/args.batch_size)
    # To avoid counting Dont count first epoch mem alloc
    for i in range(4):
        a = torch.ones(1000 * 1000 * 1000,device=i)
        del a
    forward_time_per_epoch = []
    merge_time_per_epoch = []
    data_transfer_time = []
    local_graph_splitting_time = []
    t1 = time.time()
    for i in range(args.num_epochs):
        print("epoch",i,time.time()-t1)
        for epoch_no,(in_nodes,out_nodes,blocks) in enumerate(dataloader):
            # assert(False)
            model(in_nodes, out_nodes, blocks)
            if i!=0:
                ftime, merge_time, data_movement = model.get_forward()
                forward_time_per_epoch.append(ftime)
                merge_time_per_epoch.append(merge_time)
                local_graph_splitting_time.append(model.get_graph_creation_time())
                data_transfer_time.append(data_movement)
            model.clear_timers()
    t2 = time.time()
    print("Total time :",t2 - t1)
    print("forward_time_per_epoch:{}".format(sum(forward_time_per_epoch)/(args.num_epochs - 1)))
    print("merge_time per epoch:{}".format(sum(merge_time_per_epoch)/(args.num_epochs - 1)))
    print("data transfer:{}".format(sum(data_transfer_time)/(args.num_epochs - 1)))
    print("graph splitting time:{}".format(sum(local_graph_splitting_time)/(args.num_epochs - 1)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, default = "ogbn-arxiv")
    argparser.add_argument('--fsize',type = int,default = 1024)
    argparser.add_argument('--gpu', type=str,
                            default = 0,
                            # default='0,1,2,3',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=4096)
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
