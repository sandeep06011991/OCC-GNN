
# from utils.memory_manager import unit_test_memory_manager
# unit_test_memory_manager()
# print("Memory Manager Done !!")

# from data.bipartite import unit_test_local_bipartite
# unit_test_local_bipartite()
# print("Local bipartitie done !!")

# from utils.sampler import test_sampler
# test_sampler()

def train():
    params = None
    model = get_model()
    
    print("One forward and backward pass of GCN done successfully !!")

if __name__ == "__main__":
    # Add arg parser later
    # Hardcode everything here.
    train()
# from utils.utils import get_dgl_graph
# dg_graph, workload_map = get_dgl_graph("ogbn-arxiv")
# cache_percentage = .20
# fanout = [10,10,10]
# batch_size = 1024
# mm = MemoryManager(dg_graph, dg_graph.ndata["features"], cache_percentage, \
#             fanout, batch_size, workload_map)
# Unit test graph slicer for all minibatches.
# test()
