import torch
from dgl.sampling import sample_neighbors
from data.bipartite import BipartiteGraph

class Sampler():

    def __init__(self,graph, training_nodes, workload_assignment, \
        memory_manager, fanout,batch_size):
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
        self.training_nodes = self.training_nodes[torch.randperm(self.training_nodes.shape[0])]
        self.idx = 0
        return self

    def __next__(self):
        # returns a bunch of bipartite graphs.
        if self.idx >= self.training_nodes.shape[0]:
            raise StopIteration
        batch_in = self.training_nodes[self.idx: self.idx + self.batch_size]
        blocks, layers = self.sample(batch_in)
        # Refresh last layer cache
        self.memory_manager.refresh_cache(layers[-1])
        partitioned_edges = self.edge_partitioning(blocks,layers)
        bipartite_graphs, shuffle_matrix = \
                self.create_bipartite_graphs(partitioned_edges)
        self.idx = self.idx + batch_in.shape[0]
        return bipartite_graphs, shuffle_matrix

    # Returns a dictionary of partitioned edge blocks.
    def edge_partitioning(self,blocks,layers):
        # No reordering takes place. In this part.
        no_layers = len(blocks)
        # Process except last layer.
        partitioned_blocks = []
        for layer_id in range(no_layers-1):
            # Read them reverse.
            src_ids, dest_ids = blocks[layer_id]
            partition_edges = []
            s = 0
            for gpu_id in range(4):
                selected_edges = torch.where(self.workload_assignment[src_ids] == gpu_id)[0]
                # Used for bipartite graph construction.
                s = s + selected_edges.shape[0]
                shuffle_nds = {}
                dest_ids_select_edges = dest_ids[selected_edges]
                for remote_gpu in range(4):
                    if gpu_id == remote_gpu:
                        continue
                    shuffle_nds[remote_gpu] = torch.unique(dest_ids_select_edges\
                            [torch.where(self.workload_assignment[dest_ids_select_edges] == remote_gpu)])
                partition_edges.append({"src":src_ids[selected_edges], \
                    "dest":dest_ids[selected_edges],"shuffle_nds":shuffle_nds,"device":gpu_id})
            partitioned_blocks.append(partition_edges)
            assert(s == src_ids.shape[0])
        # Select edges  What about when there is overlap
        src_ids, dest_ids  = blocks[no_layers-1]
        # All remote edges
        edge_mask = self.memory_manager.node_gpu_mask[src_ids,self.workload_assignment[dest_ids]]
        remote_edge_ids = torch.where(~ edge_mask)[0]
        natural_edge_ids = torch.where(edge_mask)[0]
        # Special edge_ids

        partition_edges = []
        for gpu_id in range(4):
            natural_edges = torch.where(self.workload_assignment[dest_ids[natural_edge_ids]]== gpu_id)[0]
            extra_edges = torch.where(self.workload_assignment[src_ids[remote_edge_ids]] == gpu_id)[0]
            src_local = torch.cat([src_ids[natural_edges], src_ids[extra_edges]])
            dest_local = torch.cat([dest_ids[natural_edges], dest_ids[extra_edges]])
            shuffle_nds = {}
            if extra_edges.shape[0]:
                for dest_gpu in range(4):
                    if dest_gpu == gpu_id:
                        continue
                    remote_edges = torch.where(self.workload_assignment[dest_ids[extra_edges]] == dest_gpu)[0]
                    shuffle_nds[dest_gpu] = torch.unique(dest_ids[extra_edges[remote_edges]])
            partition_edges.append({"src":src_local,"dest":dest_local,"shuffle_nds":shuffle_nds,"device":gpu_id})
        partitioned_blocks.append(partition_edges)
        print("Total number of edges stays the same.")
        # Correctness
        s = 0
        for i in partitioned_blocks:
            for j in i:
                s = s + j["src"].shape[0]
        ss = 0
        for src,dest in blocks:
            ss = ss + src.shape[0]
        print(ss,s)
        assert(ss == s)
        return partitioned_blocks

    def create_bipartite_graphs(self,partitioned_blocks):
        for partitioned_layer in partitioned_blocks:
            for local_layer in partitioned_layer:
                BipartiteGraph(local_layer["src"],local_layer["dest"], local_layer["device"])
        return None,None


    '''
        returns a list of blocks and edges
        Keep this agnostic to graph slicing.
        Allows me to try various sampling strategies.
    '''
    def sample(self, batch_in):
        layers = []
        blocks = []
        # Note. Dont forget self loops otherwise GAT doesnt work.
        # Create bipartite graphs for the first l-1 layers
        last_layer = batch_in
        layers.append(last_layer)
        for fanout in self.fanout:
            # Here src_ids and dest_ids are created from the point of sampler
            # Note data movement flows reverse.
            dest_ids,src_ids = sample_neighbors(self.graph, last_layer, fanout).edges()
            # Add a self loop
            self_loop_dests = torch.cat([last_layer, dest_ids])
            edges = self_loop_dests, torch.cat([last_layer, src_ids])
            last_layer = torch.unique(self_loop_dests)
            layers.append(last_layer)
            blocks.append(edges)
        return blocks,layers


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
    # MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    print("Test refresh cache misses")
    print("sampler request nodes which are currently missing")
    print("Not clear yet.")
