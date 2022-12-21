
from utils.utils import *
from utils.utils import get_process_graph
import dgl


def get_P4_data_movement(graph_name, batch_size):
    graph,_,_ = get_process_graph(graph_name, -1)
    fsize = graph.ndata['features'].shape[1]    
    dataloaders = []
    train_nid = graph.ndata['train_mask'].nonzero().flatten()
    world_size = 4
    num_hidden = 16
    for rank in range(4):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [20,20,20], replace = True)
        train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            train_nid,
            sampler,
        device='cpu',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor = 2,
        num_workers = 0)
        dataloaders.append(dataloader)
    data_moved_epoch= []
    for _ in range(5):
        data_moved = 0
        for r in range(4):
            for input_nodes,seeds,blocks in dataloaders[r]:
                data_moved += (blocks[0].number_of_dst_nodes() * num_hidden * 3)/(1024 * 1024)
        data_moved_epoch.append(data_moved)
         
    moved_data = sum(data_moved_epoch)/len(data_moved_epoch)
    with open('{}/microbench/P4.txt'.format(OUT_DIR),'a') as fp:
        fp.write("{}|{}|{}\n".format(graph_name, batch_size * 4 ,moved_data))


if __name__=="__main__":
    graph_name = "ogbn-arxiv"
    batch_size = 1024
    for graph_name in ["ogbn-products","reorder-papers100M", "amazon"]:
    #for graph_name in ["ogbn-arxiv"]:    
        for batch_size in [256,1024, 4096]: 
            get_P4_data_movement(graph_name, batch_size)
