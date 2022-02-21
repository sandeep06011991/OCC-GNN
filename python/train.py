# contains all the code for single end to end training
# primary codebase:
# Key features:
# 1. Varying models: GCN, GAT
# 2. Can turn on micro benchmarking

import argparse
import torch
import dgl
from dgl.sampling import sample_neighbors
from utils import get_dgl_graph
def micro_test():
    dg_graph,p_map =  get_dgl_graph("ogbn-arxiv")
    batch_size = 4096
    fanout = [10,10]
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        # [int(fanout) for fanout in args.fan_out.split(',')])
    import time
    dataloader = dgl.dataloading.NodeDataLoader(
        dg_graph,
        torch.arange(0,dg_graph.num_nodes()),
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    t1 = time.time()
    for i in dataloader:
        print(i)
        pass
    t2 = time.time()
    print(t2 - t1)

    t3 = time.time()
    for i in range(0,dg_graph.num_nodes(), batch_size):
        if(i+batch_size > dg_graph.num_nodes()):
            continue
        g1 = sample_neighbors(dg_graph,torch.arange(i,i+batch_size),fanout[0])
        b = torch.unique(g1.edges()[0].sort()[0])
        g2 = sample_neighbors(dg_graph,b,fanout[1])
        # print(g2)
    t4 = time.time()
    print(t4-t3)
def train(g,p_map,args):
    pass


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default = "ogbn-arxiv")
    argparser.add_argument('--fsize',type = int,default = 1024)
    # training details
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--num-workers', type=int, default=0,
       help="Number of sampling processes. Use 0 for no extra process.")

    # model name and details
    argparser.add_argument('--model-name',help="gcn|gat")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=4096)
    argparser.add_argument('--dropout', type=float, default=0.5)
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    # n_classes = 40
    # print("Hello world")
    # micro_test()
    # g,p_map =  get_dgl_graph(args.graph)
    # train(g,p_map,args)
