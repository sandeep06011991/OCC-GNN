import dgl
import torch
import time

device = 0
def get_dataset(name):
    if name =="cora":
        graphs = dgl.data.CoraFullDataset()
        dataset = graphs
    if name =="pubmed":
        graphs = dgl.data.PubmedGraphDataset()
        dataset = graphs
    if name =="reddit":
        graphs = dgl.data.RedditDataset()
        dataset = graphs
    # Returns DGLHeteroGraph
    # Create dummy dataset for testing.
    return dataset

def run_experiment(dataset_name,hops):
    dataset = get_dataset(dataset_name)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [10 for i in range(hops)])
    g = dataset[0]
    dataloader = dgl.dataloading.NodeDataLoader(
        dataset[0],
        torch.arange(g.num_nodes()),
        sampler,
        batch_size=256,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    dataloader = iter(dataloader)
    nfeat = dataset[0].ndata['feat']
    labels = dataset[0].ndata['label']
    sampling_time = 0
    formatting_time = 0
    data_movement_time = 0

    try:
        while True:
            t_a = time.time()
            input_nodes, output_nodes, blocks = next(dataloader)
            t_b = time.time()
        # for input_nodes, output_nodes, blocks in (dataloader):
            batch_inputs = nfeat[input_nodes].clone()
            batch_labels = labels[output_nodes].clone()
            t_c = time.time()
            batch_inputs.to(device)
            labels.to(device)
            block = blocks[0].int().to(device)
            torch.cuda.synchronize()
            t_d = time.time()
            sampling_time = t_b-t_a
            formatting_time = t_c - t_b
            data_movement_time = t_d - t_c
    except StopIteration:
        pass

    with open("exp1.txt","a") as fp:
        fp.write("{}|{}|{}|{}|{}\n".format(dataset_name,hops,sampling_time, \
                        formatting_time,data_movement_time))

# python3 run.py "reddit|cora|pubmed" "hops"
if __name__=="__main__":
    import sys
    assert(len(sys.argv) == 3)
    graphName = sys.argv[1]
    hops = int(sys.argv[2])
    # graphName = "cora"
    # hops = 2
    run_experiment(graphName,hops)
