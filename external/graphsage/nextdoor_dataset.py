from ctypes import *
from ctypes.util import *
import networkx as nx, numpy as np
from train_sampling import *
import torch
import time

l1 = '/mnt/homes/spolisetty/nextdoor-experiments/graph_loading/libgraph.so'
d = CDLL(l1)
d.loadgraph.argtypes = [c_char_p]
num_classes = 14
# Graph(num_nodes=232965, num_edges=114848857,
#       ndata_schemes={
        # 'label': Scheme(shape=(), dtype=torch.int64),
         # 'feat': Scheme(shape=(602,), dtype=torch.float32),
# 'test_mask': Scheme(shape=(), dtype=torch.bool),
#       'val_mask': Scheme(shape=(), dtype=torch.bool),
#       'train_mask': Scheme(shape=(), dtype=torch.bool),
#       'features': Scheme(shape=(602,), dtype=torch.float32),
#       'labels': Scheme(shape=(), dtype=torch.int64)}
#          edata_schemes={}) 41


def get_graph_from_nextdoor_datapath(filename):
    graphPath = "/mnt/homes/spolisetty/NextDoor/input/{}.data".format(filename)
    graphPath = bytes(graphPath, encoding='utf8')
    d.loadgraph(graphPath)
    print("Graph Loaded in C++")
    d.getEdgePairList.restype = np.ctypeslib.ndpointer(dtype=c_int, shape=(d.numberOfEdges(), 2))
    edges = d.getEdgePairList()
    G = nx.Graph()
    print("Loading networkx graph")
    G.add_edges_from(edges)
    N = G.number_of_nodes()
    g = dgl.from_networkx(G)
    return g,N

def fill_up_graph_data(N, G):
    G.ndata["label"] = torch.randint(0,41,(N,),device = 'cpu',requires_grad = False)
    G.ndata["feat"] = torch.rand(N, 32 ,device = 'cpu',requires_grad = False)
    G.ndata["labels"] = G.ndata["label"]
    G.ndata["features"] = G.ndata["feat"]
    b = torch.rand(N,device = 'cpu',requires_grad = False)
    G.ndata["train_mask"] = b < .8
    G.ndata["test_mask"] = (b >=.8) & (b < .9)
    G.ndata["val_mask"] = b >= .9
    return G


# G,N = get_graph_from_nextdoor_datapath()
# G = fill_up_graph_data(N,G)


def run(args, device, data,filename):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    assert(not args.sample_gpu)
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        # device=dataloader_device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    # Training time
    s = time.time()
    for epoch in range(1):
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            sum = torch.sum(input_nodes)
    e = time.time()
    print(e," ",s)
    sampling_time = e-s
    print(sampling_time)

    backward_pass_time = 0
    data_movement_time = 0
    forward_prop_time = 0
    for epoch in range(1):
        tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            s1 = time.time()
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]
            e1 = time.time()
            data_movement_time = e1 - s1 + (data_movement_time)
            # Compute loss and prediction
            s2 = time.time()
            batch_pred = model(blocks, batch_inputs)
            e2 = time.time()
            forward_prop_time = (e2-s2) + (forward_prop_time)
            s3 = time.time()
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            e3 = time.time()
            backward_pass_time += e3 - s3

    with open('results.txt','a') as fp:
        fp.write("{}|{}|{}|{}|{}\n".format(filename,sampling_time,data_movement_time,forward_prop_time,backward_pass_time))
        print("{}|{}|{}|{}|{}\n".format(filename,sampling_time,data_movement_time,forward_prop_time,backward_pass_time))



def run_experiment(filename,args):
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    ###############################################
    G,N = get_graph_from_nextdoor_datapath(filename)
    G = fill_up_graph_data(N,G)
    g = G
    n_classes = 41
    print("num_nodes",g.num_nodes())
    #################################

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')

    assert(args.data_cpu)
    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(args, device, data,filename)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=512)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true', default = False,
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true', default = True,
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()
    assert(args.data_cpu)

    with open('results.txt','a') as fp:
        fp.write("{}|{}|{}|{}|{}\n".format("dataset","sampling_time","data_movement_time","forward_prop_time","backward_pass"))


    filenames = ["ppi","reddit","patents","orkut","LJ1"]
    filenames = ["ppi","reddit","patents"]
    filenames = ["orkut","LJ1"]
    for i in filenames:
        run_experiment(i,args)
