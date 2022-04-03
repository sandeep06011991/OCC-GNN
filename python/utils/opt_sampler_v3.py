import torch
from dgl.sampling import sample_neighbors
from data.bipartite import BipartiteGraph
import time
import dgl

# Otimized Sampler which pushes batch slicing on to the gpu
class Sampler():

    def __init__(self,graph, training_nodes, workload_assignment, \
        memory_manager, fanout,batch_size):
        self.device_ids = [0,1,2,3]
        self.graph = graph
        self.training_nodes = training_nodes
        # non overlapping workload assignment
        self.workload_assignment = workload_assignment
        self.workload_assignment_gpu = [workload_assignment.to(i) for i in range(4)]
        self.batch_size = batch_size
        self.labels = graph.ndata["labels"]
        # storage assignment
        self.memory_manager = memory_manager
        self.fanout = fanout
        self.num_nodes = graph.num_nodes()
        self.sample_time = 0
        self.slice_time = 0
        self.cache_refresh_time = 0
        self.move_batch_time = 0
        self.extra_stuff = 0
        self.gpu_slice = 0
        # partition graphs
        

    def clear_timers(self):
        self.sample_time = 0
        self.slice_time = 0
        self.cache_refresh_time = 0
        self.move_batch_time = 0
        self.extra_stuff = 0
        self.gpu_slice = 0

    def __iter__(self):
        # shuffle training nodes
        self.training_nodes = self.training_nodes[torch.randperm(self.training_nodes.shape[0])]
        self.idx = 0
        return self
    # @profile
    def __next__(self):
        # returns a bunch of bipartite graphs.
        if self.idx >= self.training_nodes.shape[0]:
            raise StopIteration
        batch_in = self.training_nodes[self.idx: self.idx + self.batch_size]
        t1 = time.time()
        blocks, layers = self.sample(batch_in)
        # print("Sampling nodes",layers[0][:5])
        # print("in_degrees",self.graph.in_degrees(layers[0][:5]))
        # print("work load",self.workload_assignment[layers[0]][:5])
        # Refresh last layer cache
        t2 = time.time()
        self.memory_manager.refresh_cache(layers[-1])
        t3 = time.time()
        partitioned_edges = self.edge_partitioning(blocks,layers)
        partitioned_labels = self.partition_labels(blocks,layers)
        bipartite_graphs, shuffle_matrix, model_owned_nodes = \
                self.create_bipartite_graphs(partitioned_edges)
        t4 = time.time()
        self.sample_time += (t2-t1)
        self.slice_time += (t4-t3)
        self.cache_refresh_time += (t3-t2)
        # print("sample time", t2-t1)
        # print("cache refresh time", t3 - t2)
        # print("splitting time",t4 -t3)
        self.idx = self.idx + batch_in.shape[0]
        assert(len(bipartite_graphs) == len(shuffle_matrix))
        # Return blocks and layers for correctness.
        return bipartite_graphs, shuffle_matrix, model_owned_nodes, \
            blocks, layers, partitioned_labels

    # Blocks and layers on this gpu
    # @profile
    def edge_partitioning_gpu(self,blocks, layers, gpu_id, \
            last_layer_natural_edge, last_layer_remote_edge):
        natural_edge_ids_gpu = last_layer_natural_edge
        remote_edge_ids_gpu = last_layer_remote_edge
        no_layers = len(blocks)
        # Process except last layer.
        partitioned_blocks = []
        for layer_id in range(no_layers-1):
            # Read them reverse.
            # print(blocks[layer_id])
            src_ids, dest_ids = blocks[layer_id]
            partition_edges = None
            s = 0
            selected_edges = torch.where(self.workload_assignment_gpu[gpu_id][src_ids] == gpu_id)[0]
            # Used for bipartite graph construction.
            # s = s + selected_edges.shape[0]
            shuffle_nds = {}
            owned_nds = torch.unique(dest_ids[torch.where\
                    (self.workload_assignment_gpu[gpu_id][dest_ids]==gpu_id)[0]])
            dest_ids_select_edges = dest_ids[selected_edges]
            for remote_gpu in range(4):
                if gpu_id == remote_gpu:
                    continue
                shuffle_nds[remote_gpu] = torch.unique(dest_ids_select_edges\
                        [torch.where(self.workload_assignment_gpu[gpu_id][dest_ids_select_edges] == remote_gpu)])
            partition_edges = {"src":src_ids[selected_edges], \
                "dest":dest_ids[selected_edges],"shuffle_nds":shuffle_nds,\
                    "device":gpu_id, "owned_nds":owned_nds}
            partitioned_blocks.append(partition_edges)
            # assert(s == src_ids.shape[0])

        src_ids, dest_ids  = blocks[no_layers-1]
        natural_edges = natural_edge_ids_gpu[torch.where(\
            self.workload_assignment_gpu[gpu_id][dest_ids[natural_edge_ids_gpu]]== gpu_id)[0]]
        extra_edges = remote_edge_ids_gpu[torch.where(\
            self.workload_assignment_gpu[gpu_id][src_ids[remote_edge_ids_gpu]] == gpu_id)[0]]
        src_local = torch.cat([src_ids[natural_edges], src_ids[extra_edges]])
        # assert(torch.all(self.memory_manager.node_gpu_mask[src_local,gpu_id]))
        dest_local = torch.cat([dest_ids[natural_edges], dest_ids[extra_edges]])
        shuffle_nds = {}
        dest_nds = torch.unique(dest_ids)
        owned_nds = dest_nds[torch.where(self.workload_assignment_gpu[gpu_id][dest_nds] == gpu_id)[0]]
        if extra_edges.shape[0]:
            for dest_gpu in range(4):
                if dest_gpu == gpu_id:
                    continue
                remote_edges = torch.where( \
                    self.workload_assignment_gpu[gpu_id][dest_ids[extra_edges]] == dest_gpu)[0]
                shuffle_nds[dest_gpu] = torch.unique(dest_ids[extra_edges[remote_edges]])
        partition_edges = {"src":src_local,"dest":dest_local, \
            "owned_nds":owned_nds, "shuffle_nds":shuffle_nds,"device":gpu_id}
        partitioned_blocks.append(partition_edges)
        return partitioned_blocks

    # Returns a dictionary of partitioned edge blocks.
    # @profile
    def edge_partitioning(self,blocks,layers):
        # No reordering takes place. In this part.
        repl_blocks = []
        repl_layers = []
        t1 = time.time()
        for i in range(4):
            blocks_gpu = []
            layers_gpu = []
            for src,dest in blocks:
                blocks_gpu.append((src.to(i),dest.to(i)))
            for l in layers:
                layers_gpu.append(l.to(i))
            repl_blocks.append(blocks_gpu)
            repl_layers.append(layers_gpu)
        t2 = time.time()
        no_layers = len(blocks)
        # Select edges  What about when there is overlap
        src_ids, dest_ids  = blocks[no_layers-1]
        # All remote edges
        # Shuffle all this
        edge_mask = self.memory_manager.node_gpu_mask[src_ids,self.workload_assignment[dest_ids]]
        assert(edge_mask.shape == src_ids.shape)
        remote_edge_ids = torch.where(~ edge_mask)[0]
        natural_edge_ids = torch.where(edge_mask)[0]
        remote_edge_ids_gpus = []
        natural_edge_ids_gpus = []
        for gpu_id in range(4):
            natural_edge_ids_gpus.append(natural_edge_ids.to(gpu_id))
            remote_edge_ids_gpus.append(remote_edge_ids.to(gpu_id))
        t3 = time.time()

        partitioned_blocks_by_gpu = []
        for gpu_id in range(4):
            # blocks,layers,gpu_id, natural_edge, remote_edge
            partitioned_blocks_by_gpu.append(self.edge_partitioning_gpu(\
                repl_blocks[gpu_id], repl_blocks[gpu_id],gpu_id, natural_edge_ids_gpus[gpu_id], remote_edge_ids_gpus[gpu_id]))
        t4 = time.time()
        partitioned_blocks = []
        for layers in range(no_layers):
            partition_layer = []
            for device_id in range(4):
                partition_layer.append(partitioned_blocks_by_gpu[device_id][layers])
            partitioned_blocks.append(partition_layer)
        self.move_batch_time += (t2 - t1)
        self.extra_stuff += (t3 - t2)
        self.gpu_slice += (t4 - t3)
        # assert(torch.all(self.memory_manager.node_gpu_mask[src_ids[natural_edge_ids], \
        #                 self.workload_assignment[dest_ids[natural_edge_ids]]]))
        # # Special edge_ids
        # # Collect and reshuffle for bipartite graph construction.
        # partition_edges = []
        # for gpu_id in range(4):
        #     natural_edges = natural_edge_ids[torch.where(self.workload_assignment[dest_ids[natural_edge_ids]]== gpu_id)[0]]
        #     extra_edges = remote_edge_ids[torch.where(self.workload_assignment[src_ids[remote_edge_ids]] == gpu_id)[0]]
        #     src_local = torch.cat([src_ids[natural_edges], src_ids[extra_edges]])
        #     assert(torch.all(self.memory_manager.node_gpu_mask[src_local,gpu_id]))
        #     dest_local = torch.cat([dest_ids[natural_edges], dest_ids[extra_edges]])
        #     shuffle_nds = {}
        #     dest_nds = torch.unique(dest_ids)
        #     owned_nds = dest_nds[torch.where(self.workload_assignment[dest_nds] == gpu_id)[0]]
        #     if extra_edges.shape[0]:
        #         for dest_gpu in range(4):
        #             if dest_gpu == gpu_id:
        #                 continue
        #             remote_edges = torch.where(self.workload_assignment[dest_ids[extra_edges]] == dest_gpu)[0]
        #             shuffle_nds[dest_gpu] = torch.unique(dest_ids[extra_edges[remote_edges]])
        #     partition_edges.append({"src":src_local,"dest":dest_local, \
        #         "owned_nds":owned_nds, "shuffle_nds":shuffle_nds,"device":gpu_id})
        # partitioned_blocks.append(partition_edges)
        # # Correctness test.
        # Total number of edges across all cases must be the same.
        s = 0
        for i in partitioned_blocks:
            for j in i:
                s = s + j["src"].shape[0]
        ss = 0
        for src,dest in blocks:
            ss = ss + src.shape[0]
        # print(ss,s)
        assert(ss == s)
        return partitioned_blocks
    # @profile
    def create_bipartite_graphs(self,partitioned_blocks):
        model_graphs = []
        model_shuffles = []
        model_owned_nodes = []
        for layer in range(len(partitioned_blocks)):
            partitioned_layer = partitioned_blocks[layer]
            layer_graphs = {}
            layer_shuffles = {}
            # Fixme: Technically doest have to be another datastructure
            # Doing a quick fix for speed.
            layer_owned_nodes = {}
            total_shuffle_nodes = 0
            total_owned_nodes = 0
            for local_layer in partitioned_layer:
                src_gpu = local_layer["device"]
                if layer != len(partitioned_blocks)-1:
                    layer_graphs[src_gpu] = BipartiteGraph(local_layer["src"],local_layer["dest"], local_layer["device"])
                else:
                    layer_graphs[src_gpu] = BipartiteGraph(local_layer["src"], local_layer["dest"] \
                                    , local_layer["device"], self.memory_manager.global_to_local[:,src_gpu], \
                                     self.memory_manager.local_to_global_id[src_gpu],
                                     self.memory_manager.batch_in[src_gpu].shape[0])
                layer_shuffles[src_gpu] = local_layer["shuffle_nds"]
                layer_owned_nodes[src_gpu] = local_layer["owned_nds"]
            #     for k in local_layer["shuffle_nds"].keys():
            #         total_shuffle_nodes += local_layer["shuffle_nds"][k].shape[0]
            #     total_owned_nodes += local_layer["owned_nds"].shape[0]
            # print("Statistics of shuffle",total_shuffle_nodes/total_owned_nodes)
            model_owned_nodes.append(layer_owned_nodes)
            model_graphs.append(layer_graphs)
            model_shuffles.append(layer_shuffles)

        return model_graphs, model_shuffles, model_owned_nodes


    '''
        returns a list of blocks and edges
        Keep this agnostic to graph slicing.
        Allows me to try various sampling strategies.
    '''
    # @profile
    def sample(self, batch_in):
        layers = []
        blocks = []
        # Note. Dont forget self loops otherwise GAT doesnt work.
        # Create bipartite graphs for the first l-1 layers
        last_layer = batch_in
        last_layer = last_layer.unique()
        layers.append(last_layer)
        for fanout in self.fanout:
            # Here src_ids and dest_ids are created from the point of sampler
            # Note data movement flows reverse.
            dest_ids,src_ids = sample_neighbors(self.graph, last_layer, fanout).edges()
            # Correction test
            # Check if all edges are being sampled
            # assert((torch.where(src_ids[10] == src_ids)[0]).shape[0]
            #     == self.graph.in_degrees(src_ids[10])[0])

            # Add a self loop
            # Minibatch edge Cut
            # edge_cut_id = torch.where(self.workload_assignment[dest_ids] != self.workload_assignment[src_ids])[0]
            # nodes = torch.unique(src_ids[edge_cut_id])
            # total_nodes = torch.unique(src_ids).shape[0]
            # print(torch.unique(src_ids[edge_cut_id])[:10])
            # print(self.workload_assignment[dest_ids[edge_cut_id[:5]]])
            # print(self.graph.in_degrees(nodes).sort()[:10])
            # print("layer cut",nodes.shape[0]/total_nodes, edge_cut_id.shape[0]/dest_ids.shape[0])
            self_loop_dests = torch.cat([last_layer, dest_ids])
            edges = self_loop_dests, torch.cat([last_layer, src_ids])
            last_layer = torch.unique(self_loop_dests)
            layers.append(last_layer)
            blocks.append(edges)
        return blocks,layers

    # @profile
    def partition_labels(self, blocks, layers):
        partitioned_labels = {}
        first_layer = layers[0]
        for gpu_id in range(4):
            local_first = first_layer[torch.where(self.workload_assignment[first_layer]==gpu_id)[0]]
            partitioned_labels[gpu_id] = self.labels[local_first].to(gpu_id)
            # 2 Hop calculation used for correctness
            # self.graph.in_degrees(local_first)
            # ret = torch.zeros(local_first.shape)
            # for tgt_id in range(local_first.shape[0]):
            #     tgt = local_first[tgt_id]
            #     s = self.graph.in_degrees(tgt)[0]
            #     second_hop = torch.sum(self.graph.in_degrees(self.graph.in_edges(tgt)[0])).item()
            #     s = s + second_hop
            #     ret[tgt_id] = s
            # partitioned_labels[gpu_id] = ret
        return partitioned_labels

def test_sampler():
    print("What am I doing, design unit test 2")
    print("Takes in a graph and places all data at correct location")
    print("Test various caching percentages.!")
    from utils.utils import get_dgl_graph
    from utils.memory_manager import MemoryManager
    dg_graph,partition_map = get_dgl_graph("ogbn-arxiv")
    partition_map = partition_map.type(torch.LongTensor)
    features = torch.rand(dg_graph.num_nodes(),602)
    cache_percentage = .10
    batch_size = 1024
    fanout = [10 , 10, 10]
    mm = MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    # # (graph, training_nodes, memory_manager, fanout)
    sampler = Sampler(dg_graph, torch.arange(dg_graph.num_nodes()), partition_map, \
                mm, [10,10,10], batch_size)
    it = iter(sampler)
    next(it)
    # cache_percentage = .60
    # Memory    Manager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    print("Test refresh cache misses")
    print("sampler request nodes which are currently missing")
    print("Not clear yet.")
