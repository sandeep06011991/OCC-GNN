from utils.utils import * 
import dgl 
import dgl.nn as dglnn 
import torch.nn as nn 
from cu_train_opt import *
from torch.nn.parallel import DistributedDataParallel
from cuslicer import cuslicer 
import copy 

print("Model definitions is repeated ")
print("Refactor this ASAP ######## ")

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, \
                out_channels, heads, num_layers, activation, dropout):
        super(GAT,self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        hidden_channels = (int) (hidden_channels/heads)
        self.convs.append(dglnn.GATConv(in_channels, hidden_channels, num_heads = heads))
        for _ in range(num_layers -2):
            self.convs.append(dglnn.GATConv(hidden_channels *heads, hidden_channels,num_heads = heads))
        self.convs.append(dglnn.GATConv(hidden_channels * heads, out_channels, num_heads = 1))
        self.n_classes = out_channels
        self.n_heads = heads
        self.n_hidden = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.convs) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h = h.flatten(1)
        return h
    
    

def test_baseline_GAT(graph_name):
    device = torch.device(0)
    torch.manual_seed(0)
    graph, partition_map, n_classes, idx_split = get_dgl_graph(graph_name)
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
    batch_size = 1
    model = GAT(in_feat_dim, hidden_channels, out_channels, heads, num_layers, activation, dropout)
    
    d_graph = graph.to(device)
    train_dataloader = dgl.dataloading.DataLoader(
            d_graph,
            idx_split['train'].to(device),
            sampler,
            device=device,
            batch_size = batch_size,
            drop_last=True,
            num_workers= 0,
            shuffle=False,
            persistent_workers=False)
    
    features = features.to(0)
    model_cpu = copy.deepcopy(model)
    model = model.to(0)
    results = []
    for in_nodes, out_nodes, blocks in train_dataloader:
        out = model(blocks, features[in_nodes])
        print(out[0])
        print(in_nodes.shape)
        break
    return model_cpu, results 


def test_single_groot(proc_id, results_queue, graph_name,
                       cache_percentage, num_hidden, batch_size, \
                       exchange_queue, val_acc_queue, base_model, labels):
    torch.cuda.set_device(proc_id)
    labels = labels.to(proc_id)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = 4
    num_gpus = world_size 
    assert(world_size > 0)

    torch.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    fanout = [100000,100000]
    # Force full graph training 
    deterministic = True 
    testing = False 
    self_edge = True 
    rounds = 2 
    pull_optimization = True 
    num_layers = 2 
    use_uva = False 
    sampler = cuslicer(graph_name, cache_percentage,
                fanout ,deterministic, testing, self_edge, rounds,\
                      pull_optimization, num_layers, num_gpus, proc_id, False, use_uva)
    
    global_order_dict[Bipartite] = get_attr_order_and_offset_size(Bipartite(), num_partitions = num_gpus)
    global_order_dict[Gpu_Local_Sample] = get_attr_order_and_offset_size(Gpu_Local_Sample(), num_partitions = num_gpus)
    

    dg_graph,partition_map,num_classes = get_process_graph(graph_name, -1,-1)
    features = dg_graph.ndata['features']
    train_nid = dg_graph.ndata['train_mask'].nonzero().flatten()
    order_book = get_order_book(graph_name, cache_percentage)
    partition_offsets = get_partition_offsets(graph_name)
    gpu_local_storage  = GpuLocalStorage(cache_percentage, features, order_book, partition_offsets, proc_id)
    
    skip_shuffle = False
    model = get_gat_distributed( num_hidden, features, num_classes,
                proc_id, deterministic, "gat", pull_optimization, num_gpus,  num_layers, skip_shuffle)

    assert(len(base_model.convs) == len(model.layers))
    for lid, layer in enumerate(base_model.convs):
        model.layers[lid] = layer
    model = model.to(proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id],\
                output_device = proc_id)
                # find_unused_parameters = False
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
    optimizer = optim.Adam(model.parameters(), lr=.01)
    num_gpus = 4 
    class Args:
        pass 
    args = Args()
    args.num_hidden = num_hidden
    args.load_balance = True 
    args.batch_size = batch_size 
    attention = True 
    events = [torch.cuda.Event(enable_timing= True) for i in range(6)]
    attention = False 
    for  batch_start in torch.arange(0, train_nid.shape[0], batch_size):
        train_batch = train_nid[batch_start:batch_start + batch_size]
        # nodes = train_batch[torch.where((train_batch >= partition_offsets[proc_id]) &\
        #             (train_batch < partition_offsets[proc_id + 1]))[0]]
        target_nodes = train_batch
        isTrain = False 
        _, out = train_minibatch(target_nodes, num_gpus, partition_offsets,\
                    sampler, args, exchange_queue, optimizer, gpu_local_storage,\
                        attention, labels, events, isTrain, loss_fn, proc_id, model, val_acc_queue)
        # if proc_id == 0:
            # print(out, proc_id, batch_start)
        if (target_nodes[0].item() >= partition_offsets[proc_id]) and \
                (target_nodes[0].item() < partition_offsets[proc_id + 1]):
            print("proc_id contaisn first node")
            print(out[0][0], proc_id)
            print(out[1][0], proc_id)
            
        break    
        # results_queue.put((batch_start, proc_id , torch.sum(torch.cat(out)).item()))
    pass 


def test_groot(graph_name):
    graph, partition_map, n_classes, idx_split = get_dgl_graph(graph_name)
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
    batch_size = 1
    torch.manual_seed(0)
    model = GAT(in_feat_dim, hidden_channels, out_channels, heads, num_layers, activation, dropout)
    
    procs = []
    results_queue = mp.Queue()
    num_workers = 4
    exchange_queue = [mp.Queue(num_workers) for _ in range(num_workers)]
    val_acc_queue = mp.Queue()
    for rank in range(4):
        p = mp.Process(target = test_single_groot,\
                       args = (rank, results_queue, \
                               graph_name, "2MB", hidden_channels, batch_size,\
                                exchange_queue, val_acc_queue, model, labels))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("Analyze these results")
    print("All returned")        
#     return all_outputs and sum_of gradients 
    

if __name__ == "__main__":
    graph_name = "ogbn-arxiv"
    # mp.set_start_method('spawn')
    # test_groot("ogbn-arxiv")
    test_baseline_GAT(graph_name)