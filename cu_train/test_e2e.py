from utils.utils import * 


def test_groot(model):
    return all_outputs and sum_of gradients 
    



if __name__== "__main__":
    # Pass all the validation nodes 
    # Deterministic DGL Gat Model 2 Layer
    # Get Forwad pass expected and backward pass gradient sum
    # test multi processing 
    device = torch.device(0)

    graph, partition_map, n_classes, idx_split = get_dgl_graph("ogbn-arxiv")
    graph = graph.add_self_loop()
    labels = graph.ndata.pop('labels')
    labels = labels.type(torch.int64)
    features = graph.ndata.pop('features')
    
    in_feat_dim = features.shape[1]
    hidden_channels = 256
    out_channels = int(torch.max(labels[idx_split["train"]]).item() + 1)
    heads = 4
    num_layers = 2 
    dropout = 0
    activation = torch.nn.functional.relu

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=num_layers)
    batch_size = 4096
    from dgl_gat import GAT 
    model = GAT(in_feat_dim, hidden_channels, out_channels, heads, num_layers, activation, dropout)
    
    d_graph = graph.to(device)
    train_dataloader = dgl.dataloading.DataLoader(
            d_graph,
            idx_split.split['train'],
            sampler,
            device=device,
            batch_size = batch_size,
            drop_last=True,
            num_workers= 0,
            persistent_workers=False)
    
    for(i in )

    test()

