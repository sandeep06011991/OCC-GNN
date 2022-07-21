import torch

'''
GNN has 2 sources of data. Graph Structure and Graph feature data.
Memory manager manages the gpu memory on all devices with graph features.
Given the caching percentage and the graph, the memory manager distributes feature data
on all availble gpus.
Memory manager decides what to data to place, in which gpu and in which order.
There are 2 scearios:
    1. The cache_per per gpu < .25:
    This is a non overlapping cache, each gpu stores a non-overlapping set of vertices
    If the cache_per_gpu = .25. The total graph is partitioned across all gpus.
    If it has non overlapping cache,from its partition it chooses the high degree vertices.
    2. If the cache_per gpu >.25:
    Some vertices are replicated across gpus. The manager, for each device chooses the nodes
    in its partition along with other highest degree of remaining nodes it can fit.
After initialization:
    The manager contains which nodes are in which gpu along with their index.
    It also stores offset information, in case new nodes have to be refreshed into the cache.

Key data structures are:
    batch_in[4] : The frames of tensor used by each gpu in the forward pass of the first layer
                  If caching percentage is less than <.25. This has to be refreshed
    global_to_local: torch.ones(num_nodes,4)
    node_gpu_mask: torch bool: (num_nodes, 4)
    local_to_global[4]: of size local_sizes[i].
    offset[4]: The current number of nodes per gpu.
'''
class MemoryManager():

    # Partition map created from metis.
    def __init__(self, graph, features, class_map, cache_percentage, \
            fanout, batch_size, partition_map):
        self.graph = graph
        self.num_nodes = graph.num_nodes()
        self.features = features
        self.features.pin_memory()
        self.fsize = features.shape[1]
        assert(cache_percentage <= 1 and cache_percentage>0)
        self.cache_percentage = cache_percentage
        f = 1
        avg_degree = graph.num_edges()/graph.num_nodes()
        for ff in fanout:
            if ff == -1:
                ff = avg_degree
            f = f * ff
        self.fanout = f
        self.class_map = class_map
        self.partition_map = partition_map
        self.batch_size = batch_size
        self.clean_up = {}
        self.initialize()

    def initialize(self):
        self.batch_in = []
        self.num_nodes_cached = int(self.cache_percentage * self.graph.num_nodes() + 1)
        if(self.num_nodes_cached * 4 >= self.graph.num_nodes()):
            print("Overlapping")
        else:
            print("Non-Overlapping")
        # float is 4 bytes
        # Calculate how much space to save and reserve.
        print("GPU static cache {}:GB".format(\
                (self.num_nodes_cached * self.fsize * 4)/(1024 * 1024 * 1024)))
        # Total space is cache + expected minibatch of nodes
        self.num_nodes_alloc_per_device = int(self.fanout * 0.25 * self.batch_size) \
                            + self.num_nodes_cached
        print("GPU Allocated total space including frame {}:GB".\
            format((self.num_nodes_alloc_per_device * self.fsize * 4)/(1024 * 1024 * 1024)))
        self.local_to_global_id = []
        self.local_sizes = []
        self.check_missing = torch.zeros((self.num_nodes), dtype = torch.bool)
        self.node_gpu_mask = torch.zeros((self.num_nodes,4),dtype=torch.bool)
        self.global_to_local = torch.ones((self.num_nodes,4),dtype = torch.long) * -1
        for i in range(4):
            if self.cache_percentage <= .25:
                if(self.cache_percentage == .25):
                    # fixme: this rounding error should not happen
                    subgraph_nds = torch.where(self.partition_map == i)[0]
                    node_ids_cached = subgraph_nds
                    self.batch_in.append(torch.zeros(node_ids_cached.shape[0], self.fsize, device = i))

                else:
                    assert(False)
                    self.batch_in.append(torch.zeros(self.num_nodes_alloc_per_device, self.fsize, device = i))
                    # fixme: batch_in has different size to total number of nodes
                    # Be aware of this when constructing graphs
                    subgraph_nds = torch.where(self.partition_map == i)[0]
                    subgraph_deg = self.graph.out_degree(subgraph_nds)
                    _, indices = torch.sort(subgraph_deg,descending = True)
                    node_ids_cached = subgraph_nds[indices[:self.num_nodes_cached]]
            else:
                assert(False)
                self.batch_in.append(torch.zeros(self.num_nodes_alloc_per_device, self.fsize, device = i))
                # fixme: batch_in has different size to total number of nodes
                # Be aware of this when constructing graphs
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
            if not self.batch_in[i].is_shared():
                self.batch_in[i].detach().share_memory_()
            assert(self.batch_in[i].device == torch.device(i))
            self.check_missing = self.check_missing | self.node_gpu_mask[:,i]
        assert(torch.all(self.check_missing))
        print("No missing nodes !!")
    # @profile
    def refresh_cache(self, last_layer_nodes):
        if(self.cache_percentage >=.25):
            # Nothing to return
            return
        # Clear previously loaded vertices for consistency.
        for k in self.clean_up.keys():
            prev_nds = self.clean_up[k]
            self.global_to_local[prev_nds,k] = -1
            self.node_gpu_mask[prev_nds,k] = False
            self.local_to_global_id[k]= self.local_to_global_id[k][:-prev_nds.shape[0]]
        self.clean_up = {}
        for gpu_id in range(4):
            nodes_for_gpu = last_layer_nodes[\
                torch.where(self.partition_map[last_layer_nodes] == gpu_id)[0]]

            missing_nds = nodes_for_gpu[torch.where(self.global_to_local[nodes_for_gpu,gpu_id]==-1)[0]]
            # assert(torch.all(~self.node_gpu_mask[missing_nds,gpu_id]))
            # Fill in feature data calculating offsets
            if missing_nds.shape[0] == 0:
                continue
            # print("adding nodes", gpu_id, missing_nds)
            self.clean_up[gpu_id] = missing_nds
            off_a = self.local_sizes[gpu_id]
            off_b = missing_nds.shape[0] + off_a
            self.batch_in[gpu_id][off_a:off_b] = self.features[missing_nds]
            # assert(self.batch_in[gpu_id][:off_b] == 1)
            # assert(torch.all(self.global_to_local[missing_nds,gpu_id]==-1))
            self.global_to_local[missing_nds,gpu_id] = self.local_sizes[gpu_id] + torch.arange(missing_nds.shape[0])
            self.local_to_global_id[gpu_id] = torch.cat([self.local_to_global_id[gpu_id],missing_nds])
            self.node_gpu_mask[missing_nds,gpu_id] = True
            # assert(self.features.device == torch.device("cpu"))
            # assert(self.batch_in[gpu_id].device == torch.device(gpu_id))

def unit_test_memory_manager():
    print("unit test 1")
    print("Takes in a graph and places all data at correct location")
    print("Test various caching percentages.!")
    from utils import get_process_graph
    dg_graph,partition_map, num_classes = get_process_graph("ogbn-arxiv")
    features = torch.rand(dg_graph.num_nodes(),602)
    cache_percentage = .10
    batch_size = 1024
    fanout = [10,10,10]
    training_nodes = torch.randperm(dg_graph.num_nodes())[:1024*10]
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage,fanout, batch_size,  partition_map)
    # mm.refresh_cache(training_nodes)
    print("Memory manager complete.")

if __name__ == "__main__":
    unit_test_memory_manager()
