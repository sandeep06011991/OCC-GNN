# contains all the code for single end to end training
# primary codebase:
# Key features:
# 1. Varying models: GCN, GAT
# 2. Can turn on micro benchmarking.

import argparse
import torch
import dgl
import time
import nvtx
from dgl.sampling import sample_neighbors
from models.factory import get_model
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager
from utils.sampler import Sampler
import torch.optim as optim

def train(args):
    # Get input data
    dg_graph,partition_map,num_classes = get_process_graph(args.graph)
    #dg_graph.ndata["features"] = torch.rand(dg_graph.ndata["features"].shape[0],1024)

    partition_map = partition_map.type(torch.LongTensor)
    features = dg_graph.ndata["features"]
    features = features.pin_memory()
    cache_percentage = args.cache_per
    batch_size = args.batch_size
    fanout = args.fan_out.split(',')
    fanout = [(int(f)) for f in fanout]
    # fanout = [-1,-1]
    # Create main objects
    mm = MemoryManager(dg_graph, features, cache_percentage, \
                    fanout, batch_size,  partition_map)
    # # (graph, training_nodes, memory_manager, fanout)
    sampler = Sampler(dg_graph, torch.arange(dg_graph.num_nodes()), \
        partition_map, mm, fanout, batch_size)

    model = get_model(args.num_hidden, features, num_classes)

    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    forward_time_per_epoch = []
    cache_load_time_per_epoch = []
    graph_slice_time_per_epoch = []
    move_batch_time_per_epoch = []
    extra_stuff_per_epoch = []
    gpu_slice_per_epoch = []
    for epochs in range(args.num_epochs):
        correct = 0
        total = 0
        ii = 0
        forward_time = 0
        sampler.clear_timers()
        for b in sampler:
            ii = ii + 1
            print(ii)
            if ii > 120 and args.debug:
                break
            optimizer.zero_grad()
            bipartite_graphs, shuffle_matrices, model_owned_nodes ,\
                blocks, layers, classes = b
            bipartite_graphs.reverse()
            shuffle_matrices.reverse()
            model_owned_nodes.reverse()
            # continue
            with nvtx.annotate("forward", color="red"):
                t1 = time.time()
                outputs = model(bipartite_graphs,shuffle_matrices, \
                            model_owned_nodes, mm.batch_in)
                t2 = time.time()
                forward_time += (t2-t1)
            # print("forward_time",t2-t1)
            # if epochs%10 == 0:
            #     for  i in range(4):
            #         values, indices = torch.max(outputs[i],1)
            #         correct = correct + torch.sum(indices == classes[i]).item()
            #         total = total + classes[i].shape[0]
            # #
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
        forward_time_per_epoch.append(forward_time)
        graph_slice_time_per_epoch.append(sampler.slice_time)
        cache_load_time_per_epoch.append(sampler.cache_refresh_time)
        # move_batch_time_per_epoch.append(sampler.move_batch_time)
        # extra_stuff_per_epoch.append(sampler.extra_stuff)
        # gpu_slice_per_epoch.append(sampler.gpu_slice)
        # print("Finished one pass !!!")
        # total_loss = torch.reduce(loss(outputs,classes))
        # total_loss.backward()
        # optimizer.step()
    print("avg forward time {}".format(sum(forward_time_per_epoch[1:])/(args.num_epochs - 1)))
    print("batch slice time {}".format(sum(graph_slice_time_per_epoch[1:])/(args.num_epochs -1)))
    print("cache refresh time {}".format(sum(cache_load_time_per_epoch[1:])/(args.num_epochs - 1)))
    # print("move_batch timer per epoch {}".format(sum(move_batch_time_per_epoch[1:])/(args.num_epochs - 1)))
    # print("extra_stuff_per_epoch {}".format(sum(extra_stuff_per_epoch[1:])/(args.num_epochs - 1)))
    # print("gpu_slice_per_epoch  {}".format(sum(gpu_slice_per_epoch[1:])/(args.num_epochs - 1)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str)
    # training details
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num-workers', type=int, default=0,
       help="Number of sampling processes. Use 0 for no extra process.")

    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-per', type =float)
    argparser.add_argument('--model-name',help="gcn|gat")
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='10,10,25')
    argparser.add_argument('--batch-size', type=int, default=(1032))
    argparser.add_argument('--dropout', type=float, default=0)
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
