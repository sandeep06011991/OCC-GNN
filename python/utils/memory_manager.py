import torch
'''
Memory manager manages the gpu memory on all devices
Given the caching percentage and the graph, the memory manager distributes data
on all devices.
If it has non overlapping cache,from its partition it chooses the high degree vertices.
Otherwise, it chooses the nodes in its partition along with highest degree of remaning nodes
it can fit
Key data structures are:
    batch_in[4] : The frames of tensor used by each gpu in the forward pass of the first layer
                  If caching percentage is less than <.25. This is refreshed
    global_to_local: torch.ones(num_nodes,4)
    node_gpu_mask: torch bool: (num_nodees, 4)
    local_to_global[4]: of size local_sizes[i].

'''
class MemoryManager():

    # Partition map created from metis.
    def __init__(self, graph, features, cache_percentage, \
            fanout, batch_size, partition_map):
        self.graph = graph
        self.num_nodes = graph.num_nodes()
        self.features = features
        self.fsize = features.shape[1]
        assert(cache_percentage <= 1 and cache_percentage>0)
        self.cache_percentage = cache_percentage
        f = 1
        for ff in fanout:
            f = f * ff
        self.fanout = f
        self.partition_map = partition_map
        self.batch_size = batch_size
        self.clean_up = {}
        self.initialize()

    def initialize(self):
        self.batch_in = []
        self.num_nodes_cached = int(self.cache_percentage * self.graph.num_nodes())
        # float is 4 bytes
        print("GPU static cache {}:GB".format((self.num_nodes_cached * self.fsize * 4)/(1024 * 1024 * 1024)))
        self.num_nodes_alloc_per_device = int(self.fanout * 0.25 * self.batch_size) + self.num_nodes_cached
        print("GPU Allocated total space including frame {}:GB".format((self.num_nodes_alloc_per_device * self.fsize * 4)/(1024 * 1024 * 1024)))
        self.local_to_global_id = []
        self.local_sizes = []
        self.node_gpu_mask = torch.zeros((self.num_nodes,4),dtype=torch.bool)
        self.global_to_local = torch.ones((self.num_nodes,4),dtype = torch.long) * -1
        for i in range(4):
            self.batch_in.append(torch.zeros(self.num_nodes_alloc_per_device, self.fsize, device = i))
            if self.cache_percentage <= .25:
                subgraph_nds = torch.where(self.partition_map == i)[0]
                subgraph_deg = self.graph.out_degree(subgraph_nds)
                _, indices = torch.sort(subgraph_deg,descending = True)
                node_ids_cached = subgraph_nds[indices[:self.num_nodes_cached]]
            else:
                subgraph_nds = torch.where(self.partition_map == i)[0]
                remaining_nds = torch.where(self.partition_map != i)[0]
                remaining_deg = self.graph.out_degree(remaining_nds)
                values,indices = torch.sort(remaining_deg,descending = True)
                buffer = self.num_nodes_cached - subgraph_nds.shape[0]
                # amongst missing nodes cache them.
                additional_nds = remaining_nds[indices[:buffer]]
                node_ids_cached = torch.cat((subgraph_nds, additional_nds),dim = 0)
            self.local_to_global_id.append(node_ids_cached)
            self.local_sizes.append(node_ids_cached.shape[0])
            self.node_gpu_mask[node_ids_cached,i] = True
            self.global_to_local[node_ids_cached,i] = torch.arange(node_ids_cached.shape[0],dtype=torch.long)
            self.batch_in[i][:self.local_sizes[i]] = self.features[node_ids_cached]
            assert(self.batch_in[i].device == torch.device(i))

    def refresh_cache(self, last_layer_nodes):
        if(self.cache_percentage >=.25):
            # Nothing to return
            return
        # Clear previously loaded vertices for consistency.
        for k in self.clean_up.keys():
            prev_nds = self.clean_up[k]
            self.global_to_local[prev_nds,k] = -1

        self.clean_up = {}
        for gpu_id in range(4):
            nodes_for_gpu = last_layer_nodes[\
                torch.where(self.partition_map[last_layer_nodes] == gpu_id)]
            missing_nds = nodes_for_gpu[torch.where(self.global_to_local[nodes_for_gpu,gpu_id]==-1)[0]]
            # Fill in feature data calculating offsets
            self.clean_up[gpu_id] = missing_nds
            off_a = self.local_sizes[gpu_id]
            off_b = missing_nds.shape[0] + off_a
            self.batch_in[gpu_id][off_a:off_b] = self.features[missing_nds]
            self.global_to_local[missing_nds,gpu_id] = self.local_sizes[gpu_id] + torch.arange(missing_nds.shape[0])
            assert(self.features.device == torch.device("cpu"))
            assert(self.batch_in[gpu_id].device == torch.device(gpu_id))

def unit_test_memory_manager():
    print("unit test 1")
    print("Takes in a graph and places all data at correct location")
    print("Test various caching percentages.!")
    from utils.utils import get_dgl_graph
    dg_graph,partition_map = get_dgl_graph("ogbn-arxiv")
    features = torch.rand(dg_graph.num_nodes(),602)
    cache_percentage = .10
    last_layer_fanout = 1000
    batch_size = 1024
    fanout = [10,10,10]
    training_nodes = torch.randperm(dg_graph.num_nodes())[:1024*10]
    mm = MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    mm.refresh_cache(training_nodes)
    print("Memory manager complete.")
