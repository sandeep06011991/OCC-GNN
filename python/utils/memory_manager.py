import torch
'''
Memory manager manages the gpu memory on all devices
Given the caching percentage and the graph, the memory manager distributes data
on all devices. If it has non overlapping cache,
From its partition it chooses the high degree vertices.
Otherwise,
'''
class MemoryManager():

    # Partition map created from metis.
    def __init__(self, graph, features, cache_percentage, \
            fanout, batch_size, partition_map):
        self.graph = graph
        self.features = features
        self.fsize = features.shape[1]
        assert(cache_percentage <= 1 and cache_percentage>0)
        self.cache_percentage = cache_percentage
        self.fanout = fanout
        self.partition_map = partition_map
        self.batch_size = batch_size
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
        self.is_node_in_partition = []
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
            self.batch_in[i][:self.local_sizes[i]] = self.features[node_ids_cached]



    def refresh_cache(last_layer_tensors):
        if(self.cache_percentage >=.25):
            # Nothing to return
            return
        # Change global to local ordering
        # Update storage layer completely.
        assert(False)
        assert(self.cache_percentage < .25)

if __name__=="__main__":
    print("unit test 1")
    print("Takes in a graph and places all data at correct location")
    print("Test various caching percentages.!")
    from utils import get_dgl_graph
    dg_graph,partition_map = get_dgl_graph("ogbn-arxiv")
    features = torch.rand(dg_graph.num_nodes(),602)
    cache_percentage = .10
    last_layer_fanout = 1000
    batch_size = 1024
    fanout = 10 * 10 * 10
    MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    cache_percentage = .60
    MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    print("Test refresh cache misses")
    print("sampler request nodes which are currently missing")
    print("Not clear yet.")
