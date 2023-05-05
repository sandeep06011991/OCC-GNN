from utils.utils import get_process_graph
import dgl
import torch
import time
import nvtx
import numpy as np

def new_partition_load(partitions: torch.Tensor, degree: torch.Tensor ) -> torch.Tensor:
    ld = []
    for i in range(4):
        ld.append(torch.sum(partitions == i))
    
    balanced = int(sum(ld)/4)
    for i in range(4):
        if ld[i] < balanced:
            continue
        # Find an unbalanced partition
        for j in range(4):
            if i == j or ld[j] >= balanced:
                continue
            if ld[i] == balanced:
                break
            to_add = ld[i] - balanced
            can_add = min(int( 1.1 * (balanced - ld[j])) , to_add)
            assert(ld[j] < balanced)

            current = torch.where(partitions == i)[0]
            #add_position = current[torch.argsort(degree[current], descending = False)][:can_add]
            add_position = current[torch.randperm(current.shape[0])][:can_add] 


            assert(torch.all(partitions[add_position] == i))
            partitions[add_position] = j
            ld[j] += can_add
            ld[i] -= can_add
            print("Added", can_add, "from", i, "to", j)
    print(ld)
    print("Partitioning complete")
    return partitions

def partition_load(edges, partition, k):
    load = []
    all_edges = []
    load_e_local = []
    load_e_pulled = []
    total_nodes = partition.shape[0]
    total_edges = edges[0].shape[0]
    shuffle_volume = []
    for i in range(k):
        load.append(torch.sum(partition == i)) #/total_nodes)
        load_e_local.append(\
                torch.sum((partition[edges[1]] == i)  & (partition[edges[0]] == i)))#/total_edges)
        load_e_pulled.append(\
                torch.sum((partition[edges[1]] == i)  & (partition[edges[0]] != i)))
        all_edges.append(torch.sum((partition[edges[1]] == i)))
        #print(edges[torch.where((partition[edges[1]] == i)  & (partition[edges[0]] != i))[0])
        shuffle_volume.append(torch.unique(edges[0][torch.where((partition[edges[1]] == i)  & (partition[edges[0]] != i))]).shape[0])
        shuffle_volume[-1] = (shuffle_volume[-1] * 4 * 256 )/(1024 ** 2)
    print("shuffle volume MB:", shuffle_volume)
    print("load_e_local ", load_e_local)
    print("load_e_pulled", load_e_pulled)
    print("edges_all", all_edges)
    print("node",load)
    print("node skew", (max(load) - min(load))/min(load))
    print("edge skew", (max(all_edges) - min(all_edges))/min(all_edges))
    # Find heaviy loaded partition



def edges_per_epoch(filename, batch_size):
    device = 'cpu'
    #torch.cuda.set_device(device)
    # filename = "ogbn-arxiv"
    
    num_gpus = 4

    dg_graph, partition_map, num_classes =\
        get_process_graph(filename, -1 ,  num_gpus)
    fsize = dg_graph.ndata['features'].shape[1]
    g = dg_graph.to(device)
    p_map = partition_map
    e1, e2 = g.edges()
    #tnds = g.ndata['train_idx'].nonzero()
    print("Training node division")
    #partition = dgl.metis_partition(dg_graph,4, balance_edges = True, reshuffle = False)
    
    #p_map = torch.zeros((g.num_nodes(),))
    #for k in partition.keys():
    #    p_map[partition[k].ndata['_ID']] = k    

    #with open('edge_balance_pr.bin','wb') as fp:
    #    fp.write(p_map.numpy().astype(np.intc).tobytes())

    print("Node division")
    for i in range(4):
        print(i, torch.sum(p_map == i))
    

    print("Edege Division")
    for i in range(4):
        for j in range(4):
            print(i, j, torch.sum((p_map[e1] == i) & (p_map[e2] == j)))
    #partition = dgl.metis_partition(dg_graph,4, balance_edges = True, reshuffle = False)
    #
    #p_map = torch.zeros((g.num_nodes(),))
    #for k in partition.keys():
    #    p_map[partition[k].ndata['_ID']] = k

    print(" Device")
    #print("number of nodes", g.num_nodes())
    #train_nid = torch.arange(0,g.num_nodes(), device = device)
    train_nid = g.ndata['train_mask'].nonzero().flatten()
    print(len(train_nid)/batch_size,"total number of batches")

    num_neighbors = 20
    # sampler = dgl.dataloading.NeighborSampler(
    #     [15,10,5], replace = True)

    sampler = dgl.dataloading.NeighborSampler(
        [int(num_neighbors) for i in range(3)], replace = False)
    dataloader = dgl.dataloading.NodeDataLoader(\
        g,\
        train_nid,\
        sampler,\
        device=device,\
        batch_size=batch_size,\
        shuffle=True,\
        drop_last=True)
    dt = iter(dataloader)
    i = 0
    t1 = time.time()
    total_edges = 0
    j = 0
    degree = g.in_degrees()
    try:
        while True:
            t1 = time.time()
            with nvtx.annotate("Sample",color = 'blue'):
                input_, output, blocks = next(dt)
                edges = []
                nodes = []
                nodes.append(input_.shape[0])
                #print(blocks[0].edges())
                #src, dest = blocks[0].edges()
                #src, dest = input_[src], input_[dest]
                '''
                if j < 5:
                    new_g = dgl.DGLGraph((src,dest))
                    check = dgl.metis_partition(new_g, 4, balance_edges = True)
                    for i in check.keys():
                        in_same_partition = check[i].ndata['_ID']
                        same = []
                        cur = 0
                        cur_p = -1
                        for j in range(4):
                            t = (torch.sum(p_map[in_same_partition] == j))
                            same.append(t)
                            if t > cur:
                                cur_p = j
                                cur = t
                        p_map[in_same_partition] = cur_p
                    j  = j + 1
                    #print(i, same, cur_p)    
                '''
                if batch_size != 1024:
                    new = new_partition_load(p_map[input_].clone(), degree[input_])
                
                    print("Before load balancing")
                    partition_load( blocks[0].edges(), p_map[input_], 4)
                    print("Load after Load balancing")
                    partition_load( blocks[0].edges(), new, 4)
                for b in blocks:
                    edges.append(b.num_edges())
                    nodes.append(b.num_dst_nodes())
                    total_edges += (b.num_edges())
                print("Nodes", nodes , "Edges", edges)
            i = i + 1
            if(i>10):
                break
            t2 = time.time()
            # print("Total sampling time", t2-t1)
    except StopIteration:
        pass
    return total_edges


# Ogbn-products with a fanout of 20 and three layers
# bs 1024 edges 8319487800
# bs 4096 edges 6621680480 0.7959240567670525
# bs 16384 edges 3979739620 0.601016559470112
def test_redundancy_in_edges():
    graph_name = "reorder-papers100M"
    #reorder-papers100M"
    #e1 = edges_per_epoch(graph_name, 1024)

    e2 = edges_per_epoch(graph_name, 4096)

    #e3 = edges_per_epoch(graph_name, 4096)

    print("bs",1024, "edges", e1)
    print("bs",4*1024, "edges", e2, e2/e1)
    #print("bs",4*4 *1024, "edges", e3, e3/e2)

# b = test_sampling_overhead()
if __name__=="__main__":
    t = test_redundancy_in_edges()
    
