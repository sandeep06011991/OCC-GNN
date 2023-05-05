from utils.utils import get_process_graph
import dgl
import torch
import time
import nvtx

def test_sampling_overhead():
    torch.cuda.set_device(0)
    filename = "ogbn-products"

    fsize = 128
    num_gpus = 4
    dg_graph, partition_map, num_classes =\
        get_process_graph(filename, -1 ,  num_gpus)
    g = dg_graph.to(0)
    print("number of nodes", g.num_nodes())
    train_nid = torch.arange(0,int(g.num_nodes()/4), device = torch.device(0))
    # train_nid = g.ndata['train_mask'].nonzero().flatten()
    
    print(len(train_nid)/4096,"total number of batches")
    num_neighbors = 20
    sampler = dgl.dataloading.NeighborSampler(
        [int(num_neighbors) for i in range(3)], replace = True)
    sampler = dgl.dataloading.NeighborSampler(
            [15,10,5], replace = True)

    batch_size = 4096
    dataloader = dgl.dataloading.NodeDataLoader(\
        g,\
        train_nid,\
        sampler,\
        device='cuda',\
        batch_size=batch_size,\
        shuffle=False,\
        drop_last=True)
    dt = iter(dataloader)
    i = 0
    total_time = 0
    s2 = torch.cuda.Stream()
    try:
        while True:
            t1 = time.time()
            with nvtx.annotate("Sample",color = 'blue'):
                with torch.cuda.stream(s2):
                    b = (next(dt))
                    for i in range(3):
                        n = b[2][i].number_of_dst_nodes() 
                        e = b[2][i].number_of_edges()
                        print("Avg degree", e/n)
            t2 = time.time()
            total_time += t2-t1
            #print("Total sampling time", t2-t1)

    except StopIteration:
        pass
    print("Total time", total_time)


#b = test_sampling_overhead()
if __name__=="__main__":
    b = test_sampling_overhead()
