import torch
from cslicer import cslicer
from utils.utils import get_process_graph
# Test 0%, 25% and 100%
def run_gcn_test():
    graphname = "ogbn-products"
    graph, p_map , _ = get_process_graph(graphname ,-1)
    N = (graph.num_nodes())
    training_nds = graph.ndata['train_mask'].nonzero().flatten()
    batch_size = 4096
    cache_per = {
                # 0: [[],[],[],[]],
                 25: [torch.where(p_map==i)[0].tolist() for i in range(4)],
                # 100: [[i for i in range(N)] for _ in range(4)]
                }
    for cache in cache_per.keys():
        slicer = cslicer(graphname ,cache_per[cache], 20, False, False, False)
        i = 0
        while(i < training_nds.shape[0]):
            nds = training_nds[i:i+batch_size].tolist()
            print("verification tool",slicer.sampleAndVerify(nds))
            i = i + batch_size




if __name__=="__main__":
    run_gcn_test()
