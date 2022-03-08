# contains all the code for single end to end training
# primary codebase:
# Key features:
# 1. Varying models: GCN, GAT
# 2. Can turn on micro benchmarking

import argparse
import torch
import dgl
from dgl.sampling import sample_neighbors
from utils.utils import get_dgl_graph
from models.factory import get_model
from utils.utils import get_dgl_graph
from utils.memory_manager import MemoryManager
from utils.sampler import Sampler
import torch.optim as optim

def train(args):
    # Get input data
    dg_graph,partition_map = get_dgl_graph("ogbn-arxiv")
    partition_map = partition_map.type(torch.LongTensor)
    features = dg_graph.ndata["features"]
    cache_percentage = 1
    batch_size = 10000
    fanout = [10, 10, 25]
    # Create main objects
    mm = MemoryManager(dg_graph, features, cache_percentage,fanout, batch_size,  partition_map)
    # # (graph, training_nodes, memory_manager, fanout)
    sampler = Sampler(dg_graph, torch.arange(dg_graph.num_nodes()), partition_map, \
                mm, fanout, batch_size)
    model = get_model(features, dg_graph.ndata["labels"])
    print("Model constructed")
    loss = torch.nn.CrossEntropyLoss()
    for m in (model.parameters()):
        print(m.shape)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for epochs in range(100):
        correct = 0
        total = 0
        for b in sampler:
            optimizer.zero_grad()
            bipartite_graphs, shuffle_matrices, model_owned_nodes , blocks, layers, classes = b
            bipartite_graphs.reverse()
            shuffle_matrices.reverse()
            model_owned_nodes.reverse()
            outputs = model(bipartite_graphs,shuffle_matrices, \
                        model_owned_nodes, mm.batch_in)
            if epochs%10 == 0:
                for  i in range(4):
                    values, indices = torch.max(outputs[i],1)
                    correct = correct + torch.sum(indices == classes[i]).item()
                    total = total + classes[i].shape[0]

            losses = []
            for i in range(4):
                losses.append(loss(outputs[i],classes[i]))
            loss_gather = torch.nn.parallel.gather(losses,0)
            total_loss = torch.sum(loss_gather,0)
            # print("Total loss is ", total_loss)
            total_loss.backward()
            optimizer.step()
        if epochs%10 ==0:
            print("Accuracy", correct, total)

        # print("Finished one pass !!!")
        # total_loss = torch.reduce(loss(outputs,classes))
        # total_loss.backward()
        # optimizer.step()

    print("sampler gives all the data required Done !!")
    print("model forward pass is working (Not Done)")
    print("model back pass and overall accuracy (Not Done)")
    print("What are the other things that I need")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default = "reddit")
    argparser.add_argument('--fsize',type = int,default = 602)
    # training details
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num-workers', type=int, default=0,
       help="Number of sampling processes. Use 0 for no extra process.")

    # model name and details
    argparser.add_argument('--model-name',help="gcn|gat")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=8,
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
    train(args)
