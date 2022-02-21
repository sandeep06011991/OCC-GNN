import torch
from dgl.sampling import sample_neighbors
from data.bipartite import BipartiteGraph

class Sampler():

    def __init__(self,graph, training_nodes, workload_assignment, memory_manager, fanout):
        self.graph = graph
        self.training_nodes = training_nodes
        # non overlapping workload assignment
        self.workload_assignment = workload_assignment
        self.batch_size = batch_size
        # storage assignment
        self.memory_manager = memory_manager
        self.fanout = fanout
        self.num_nodes = graph.num_nodes()

    def __iter__(self):
        # shuffle training nodes
        self.training_nodes = torch.randperm(self.training_nodes)
        self.idx = 0
        return self

    def __next__(self):
        # returns a bunch of bipartite graphs.
        if self.idx >= self.training_nodes.shape[0]:
            raise StopIteration
        batch_nodes = self.training_nodes[self.idx + self.batch_size]
        blocks, layers = self.sample(batch_in)
        # Refresh last layer cache
        self.memory_manager.refresh(layers[-1])
        partitioned_edges = self.edge_partitioning(blocks,layers)
        bipartite_graphs, shuffle_matrix = \
                self.create_bipartite_graphs(partitioned_edges)
        self.idx = self.idx + batch_nodes.shape[0]
        return bipartite_graphs, shuffle_matrices

    # Returns a dictionary of partitioned edge blocks.
    def edge_partitioning(blocks,layers):
        # No reordering takes place. In this part.
        no_layers = len(blocks)
        # Process except last layer.
        partitioned_blocks = []
        for layer_id in range(no_layers-1):
            # Read them reverse.
            src_ids, dest_ids = blocks[layer_id]
            partitoned_edges = []
            for gpu_id in range(4):
                selected_edges = torch.where(self.workload_assignment[src_ids] == gpu_id)
                # Used for bipartite graph construction.
                shuffle_nds = {}
                dest_select_edges = dest_ids[selected_edges]
                for remote_gpu in range(4):
                    if gpu_id == remote_gpu:
                        continue
                    shuffle_nds[remote_gpu] = torch.unique(dest_select_edges\
                            [torch.where(self.workload_assignment[dest_select_edges[dest_ids]] == remote_gpu)])
                partition_edges.append({"src":src_ids[selected_edges], \
                    "dest":dest_ids[selected_edges],"shuffle_nds":shuffle_nds})
            partitioned_blocks.append(partitioned_layers)
        # Select edges  What about when there is overlap
        src_ids, dest_ids  = blocks[no_layers-1]
        # All remote edges
        edge_ids = torch.where(not self.storage_layer[src_ids,partition_map[dest_ids]])
        # Special edge_ids
        shuffle_nds = {}
        for gpu_id in range(4):
            natural_edges = torch.where(dest_id[partition_map] == gpu_id)
            extra_edges = torch.where(partition_map[src_ids[edge_ids]] == gpu_id)
            src_ids = torch.concat([src_ids[natural_edges], src_ids[edge_ids[extra_edges]]])
            dest_ids = torch.concat([dest_ids[natural_edges], src_ids[edge_ids[extra_edge]]])
            if not natural_edges.shape[0]:
                for dest_gpu in range(4):
                    if dest_gpu == gpu_id:
                        continue
                    remote_edges = torch.where(self.workload_map[extra_edge[dest_ids]] == dest_gpu)
                    shuffle_nds[dest_gpu] = torch.unique(dest_ids[extra_edge[remote_edges]])
            partition_edges.append({"src":src_ids,"dest":dest_ids,"shuffle_nds":shuffle_nds})
        partitioned_blocks.append(partitioned_edges)
        print("Total number of edges stays the same.")
        # Correctness
        s = 0
        for i in partitioned_blocks:
            for j in i:
                s = s + j["src_ids"].shape[0]
        ss = 0
        for src,dest in blocks:
            ss = ss + src.shape[0]
        assert(ss == s)
        return partitioned_blocks

    def create_bipartite_graphs(partitioned_blocks):
        pass
    # def graph_slicing_without_overlap(self,blocks,layers):
    #     reordered_tensors = []
    #     for layer in layers:
    #         # Except 1
    #         reordered_tensors.append(DistTensor(layer,
    #                 self.memory_manager.partition_map[layer], self.num_nodes))
    #     bipartite_graphs = []
    #     for  block_id in len(blocks):
    #         # Reordering is done in reverse as well.
    #         dist_map_in = reordered_tensors[block_id+1]
    #         dist_map_out = reordered_tensors[block_id]
    #         # Bipartite graphs are processed and constructed in reverse.
    #         src_nds, dest_nds = blocks[block_id]
    #         pid_src_nds, pid_dest_nds = dist_map_in.global_to_gpu_id[src_nds], dist_map_out.global_to_gpu_id[dest_nds]
    #         layer_graphs = []
    #         shuffle_matrices []
    #         for src_gpu in range(4):
    #             # Three kinds of edges in this graph
    #             # src-src, src-remote.
    #             # Find nodes local
    #             # Step1 : Collect all edges while keeping global ordering.
    #             edges_with_src = torch.where(pid_src_nds == src_gpu)[0]
    #             dest_nds_for_src = dest_nds[edges_with_src]
    #             partition_dest_edges_for_src = pid_dest_nds[edge_with_src]
    #             edge_with_src_remote_dest = torch.where(partition_dest_edges_for_src != src_gpu)[0]
    #             edge_with_src_dest = torch.where(partition_dest_edges_for_src == src_gpu))[0]
    #             # For renumbering and create bipartite graph
    #             dest_local_nodes = torch.zeros(edge_with_src.shape[0], dtype = torch.int)
    #             src_local_nodes = dist_map_in.global_to_local_id[src_nds[edge_with_src]]
    #             assert(src_local_nodes.shape == dest_local_nodes.shape)
    #             dest_local_nodes[edge_with_src_dest] = dist_map_out.global_to_local_id[dest_nds_for_src[edge_with_src_dest]]
    #             remote_dest_nodes = dest_nds_for_src[edge_with_src_remote_dest]
    #             no_dup,mapping = remote_dest_nodes.unique(return_inverse = True)
    #             num_local_dest_nodes = dist_map_out.local_sizes[src_gpu]
    #             mapping = mapping + num_local_dest_nodes
    #             dest_local_edges[edge_with_src_remote_dest] = mapping.type(torch.int32)
    #             num_dest_nodes = no_dup.shape[0] + dist_map_out.local_sizes[src_gpu]
    #             num_src_nodes = dist_map_in.local_sizes[src_gpu]
    #             graphs.append((src_gpu, BipartiteGraph(src_local_nodes, dest_local_edges,num_src_nodes, num_dest_nodes))
    #             shuffle_matrix = []
    #             for dest_gpu in range(4):
    #                 if src_gpu == dest_gpu:
    #                     continue
    #                     from_id = torch.where(dist_map_out.global_to_gpu_id[no_dup == dest_gpu)
    #                     from_id = from_src  + num_local_dest_nodes
    #                     to_id = dist_map_out.global_to_gpu_id[no_dup[from_id]]
    #                     shuffle_matrix.append((src_gpu, dest_gpu, from_id, dest_id))
    #     # Add overlap
    #     returns bipartite_graphs, shuffle_matrices, cache_refresh_map


    # def graph_slicing_with_overlap(self,blocks,layers):
    #     # This code is copied from above.
    #     # Rethink nomenclature and fix this later.
    #     reordered_tensors = []
    #     for layer_id in len(layers)-1:
    #         # Except 1
    #         reordered_tensors.append(DistTensor(layer,
    #                 self.memory_manager.partition_map[layer], self.num_nodes))
    #     bipartite_graphs = []
    #     for  block_id in len(blocks):
    #         # Reordering is done in reverse as well.
    #         dist_map_in = reordered_tensors[block_id+1]
    #         dist_map_out = reordered_tensors[block_id]
    #         # Bipartite graphs are processed and constructed in reverse.
    #         src_nds, dest_nds = blocks[block_id]
    #         pid_src_nds, pid_dest_nds = dist_map_in.global_to_gpu_id[src_nds], dist_map_out.global_to_gpu_id[dest_nds]
    #         layer_graphs = []
    #         shuffle_matrices []
    #         for src_gpu in range(4):
    #             # Three kinds of edges in this graph
    #             # src-src, src-remote.
    #             # Find nodes local
    #             # Step1 : Collect all edges while keeping global ordering.
    #             edges_with_src = torch.where(pid_src_nds == src_gpu)[0]
    #             dest_nds_for_src = dest_nds[edges_with_src]
    #             partition_dest_edges_for_src = pid_dest_nds[edge_with_src]
    #             edge_with_src_remote_dest = torch.where(partition_dest_edges_for_src != src_gpu)[0]
    #             edge_with_src_dest = torch.where(partition_dest_edges_for_src == src_gpu))[0]
    #             # For renumbering and create bipartite graph
    #             dest_local_nodes = torch.zeros(edge_with_src.shape[0], dtype = torch.int)
    #             src_local_nodes = dist_map_in.global_to_local_id[src_nds[edge_with_src]]
    #             assert(src_local_nodes.shape == dest_local_nodes.shape)
    #             dest_local_nodes[edge_with_src_dest] = dist_map_out.global_to_local_id[dest_nds_for_src[edge_with_src_dest]]
    #             remote_dest_nodes = dest_nds_for_src[edge_with_src_remote_dest]
    #             no_dup,mapping = remote_dest_nodes.unique(return_inverse = True)
    #             num_local_dest_nodes = dist_map_out.local_sizes[src_gpu]
    #             mapping = mapping + num_local_dest_nodes
    #             dest_local_edges[edge_with_src_remote_dest] = mapping.type(torch.int32)
    #             num_dest_nodes = no_dup.shape[0] + dist_map_out.local_sizes[src_gpu]
    #             num_src_nodes = dist_map_in.local_sizes[src_gpu]
    #             graphs.append((src_gpu, BipartiteGraph(src_local_nodes, dest_local_edges,num_src_nodes, num_dest_nodes))
    #             shuffle_matrix = []
    #             for dest_gpu in range(4):
    #                 if src_gpu == dest_gpu:
    #                     continue
    #                     from_id = torch.where(dist_map_out.global_to_gpu_id[no_dup == dest_gpu)
    #                     from_id = from_src  + num_local_dest_nodes
    #                     to_id = dist_map_out.global_to_gpu_id[no_dup[from_id]]
    #                     shuffle_matrix.append((src_gpu, dest_gpu, from_id, dest_id))
    #     # Add overlap
    #
    #     returns bipartite_graphs, shuffle_matrices, cache_refresh_map


    '''
        returns a list of blocks and edges
        Keep this agnostic to graph slicing.
        Allows me to try various sampling strategies.
    '''
    def sampling(self, batch_in):
        layers = []
        blocks = []
        # Note. Dont forget self loops otherwise GAT doesnt work.
        # Create bipartite graphs for the first l-1 layers
        layers.append(last_layer)
        for i in len(self.fanouts):
            fanout = self.fanouts[i]
            # Here src_ids and dest_ids are created from the point of sampler
            # Note data movement flows reverse.
            dest_ids,src_ids = sample_neighbors(self.dg_graph, last_layer, fanout)
            # Add a self loop
            self_loop_dests = torch.concat([last_layer, dest_ids])
            last_layer = torch.unique(self_loop_dests)
            edges = last_layer, torch.concat([last_layer, src_ids])
            layer.append(last_layer)
            blocks.append(edges)
        return blocks,layers


def test():
    print("What am I doing, design unit test 2")
    print("Takes in a graph and places all data at correct location")
    print("Test various caching percentages.!")
    from utils.utils import get_dgl_graph
    from utils.memory_manager import MemoryManager
    dg_graph,partition_map = get_dgl_graph("ogbn-arxiv")
    features = torch.rand(dg_graph.num_nodes(),602)
    cache_percentage = .10
    last_layer_fanout = 1000
    batch_size = 1024
    fanout = 10 * 10 * 10
    mm = MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    # # (graph, training_nodes, memory_manager, fanout)
    sampler = Sampler(dg_graph, torch.arange(dg_graph.num_nodes()), mm, [10,10,10])
    it = iter(sampler)
    next(it)
    # cache_percentage = .60
    # MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    print("Test refresh cache misses")
    print("sampler request nodes which are currently missing")
    print("Not clear yet.")
