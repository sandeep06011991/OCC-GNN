# Reaches around 0.7866 ± 0.0041 test accuracy.
import logging
import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from dgl_sage import SAGE
from dgl_gat import GAT

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
        PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
        PATH_DIR = "/home/spolisetty/OCC-GNN"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
        PATH_DIR = "/home/q91/OCC-GNN"
    return DATA_DIR,PATH_DIR

DATA_DIR, PATH_DIR = get_data_dir()
def average(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)


# Entry point

def run(rank, args,  data):
    # Unpack data

    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, offsets, test_graph = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')], replace = True)
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
    number_of_minibatches = train_nid.shape[0]/args.batch_size
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cpu',
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor = 0,
        num_workers=0,
        persistent_workers = 0)
        # blocks.
     for i in range(5):
        dataloader_i = iter(dataloader)
        try:
            while True:
                input_nodes, seeds, blocks = next(dataloader_i)
                blocks = [blk.to(device) for blk in blocks]
                # Measure exapnsion factor. 
                #t2 = time.time()
                blocks = [blk.formats(['coo','csr','csc']) for blk in blocks]

                for blk in blocks:
                    blk.create_formats_()
                    edges_computed += blk.edges()[0].shape[0]
                #print(edges_computed)
                t3 = time.time()
                #start = offsets[device][0]
                #end = offsets[device][1]
                #hit = torch.where((input_nodes > start) & (input_nodes < end))[0].shape[0]

                #hit = torch.where(input_nodes < offsets[3])[0].shape[0]
                # Compute loss and prediction

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default = 0)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--sample-gpu', action='store_true')
    argparser.add_argument('--early-stopping', action = 'store_true')
    argparser.add_argument('--test-graph',type = str)
    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load ogbn-products data
    #root = "/mnt/bigdata/sandeep/"
    import utils
    dg_graph, partition_map, num_classes = utils.get_process_graph(args.graph)
    data = dg_graph
    train_idx = torch.where(data.ndata.pop('train_mask'))[0]
    val_idx = torch.where(data.ndata.pop('val_mask'))[0]
    test_idx = torch.where(data.ndata.pop('test_mask'))[0]
    graph = dg_graph
    labels = dg_graph.ndata.pop('labels')
    nfeat = dg_graph.ndata.pop('features')
    graph = graph.add_self_loop()
    test_graph = None
    if (args.test_graph) != None:
        test_graph, _, num_classes = utils.get_process_graph(args.test_graph, True)
        test_graph = test_graph.add_self_loop()
    offsets = {3:0}
    ###################################
    #data = DglNodePropPredDataset(name=args.graph, root=root)
    #splitted_idx = data.get_idx_split()
    #train_idx, val_idx, test_idx = data.ndata['train_idx'], splitted_idx['valid'], splitted_idx['test']
    #graph, labels = data[0]
    #graph = graph.add_self_loop()
    #labels = labels[:, 0].to(device)

    # feat = graph.ndata.pop('feat')
    #year = graph.ndata.pop('year')

    in_feats = nfeat.shape[1]
    nfeat = nfeat.pin_memory()
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, offsets, test_graph

    #test_accs = []
    #for i in range(1, 11):
    #    print(f'\nRun {i:02d}:\n')
    #    test_acc = run(args, device, data)
    #    test_accs.append(test_acc)
    #test_accs = th.tensor(test_accs)
    print('============================')
    #print(f'Final Test: {test_accs.mean():.4f} ± {test_accs.std():.4f}')
    world_size = 4
    mp.spawn(
        run,
        args=(args,  data),
        nprocs=world_size,
        join=True
    )
