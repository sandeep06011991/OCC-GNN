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
from cslicer import cslicer
from data.compr_cbipartite import Bipartite, Sample
from queue import Queue
import threading
import os
os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/cslicer/"

def train(args):
    # Get input data
    dg_graph,partition_map,num_classes = get_process_graph(args.graph, args.fsize)
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
    mm = MemoryManager(dg_graph, features, num_classes, cache_percentage, \
                    fanout, batch_size,  partition_map)
    #(graph, training_nodes, memory_manager, fanout)
    queue_size = 256
    no_worker_threads = 32
    number_of_epochs = args.num_epochs
    minibatch_size = args.batch_size
    storage_vector = []
    for i in range(4):
        storage_vector.append(mm.local_to_global_id[i].tolist())
    assert(len(storage_vector) == 4)
    sampler = cslicer(args.graph, queue_size, no_worker_threads, number_of_epochs, minibatch_size, storage_vector)
    # sampler = Sampler(dg_graph, torch.arange(dg_graph.num_nodes()), \
    #     partition_map, mm, fanout, batch_size)
    time.sleep(10)

    #model = get_model(args.num_hidden, features, num_classes)

    loss = torch.nn.CrossEntropyLoss()

    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    forward_time_per_epoch = []
    cache_load_time_per_epoch = []
    graph_slice_time_per_epoch = []
    move_batch_time_per_epoch = []
    extra_stuff_per_epoch = []
    gpu_slice_per_epoch = []
    total_time_per_epoch = []
    minibatches_per_epoch = int(sampler.getNoSamples()/args.num_epochs)
    print("minibatches per epoch ", minibatches_per_epoch)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for epochs in range(args.num_epochs):
        correct = 0
        total = 0
        ii = 0
        forward_time = 0
        slice_time = 0
        # fixme: iterate through all samples
        t01 = time.time()
        def prefetch(queue,sampler,minibatches_per_epoch):
            for b in range(minibatches_per_epoch):
                csample = sampler.getSample()
                tensorized_sample = Sample(csample)
                queue.put(tensorized_sample)
        q = Queue(3)
        th = threading.Thread(target = prefetch, args = (q,sampler,minibatches_per_epoch))
        th.start()
        for b in range(minibatches_per_epoch):
            ii = ii + 1
            # print("minibatch",ii)
            # if ii > 120 and args.debug:
            #    break
            #optimizer.zero_grad()
            t1 = time.time()
            # csample = sampler.getSample()
            t3 = time.time()
            # tensorized_sample = Sample(csample)
            # print("Attempting pop")
            tensorized_sample = q.get()
            # assert(False)
            continue
            t2 = time.time()
            # print("time to pop sample",t3-t1)
            # print("time to tensoerize",t2-t3)
            slice_time += (t2-t1)
            # continue
            # with nvtx.annotate("forward", color="red"):
            # if True:
            t1 = time.time()
            # torch.cuda.set_device(0)
            # start.record()
            outputs = model(tensorized_sample.layers, mm.batch_in)
           #     t2 = time.time()
           #     forward_time += (t2-t1)
           #     print("Forward time", t2-t1)
           # print("Forward Pass successful !!")
            classes = []
            for gpu in range(4):
                gpu_nodes = tensorized_sample.last_layer_nodes[gpu]
                classes.append(dg_graph.ndata["labels"][gpu_nodes].to(torch.device(gpu)))

            # print("forward_time",t2-t1)
            # if epochs%10 == 0:
            #     for  i in range(4):
            #         values, indices = torch.max(outputs[i],1)
            #         correct = correct + torch.sum(indices == classes[i]).item()
            #         total = total + classes[i].shape[0]
            # #

            losses = []
            for i in range(4):
                #print(outputs[i].shape, classes[i].shape)
                losses.append(loss(outputs[i],classes[i]))
            loss_gather = torch.nn.parallel.gather(losses,0)
            total_loss = torch.sum(loss_gather,0)
            # print("Total loss is ", total_loss)
            total_loss.backward()
            # end.record()
            # torch.cuda.set_device(0)

            t2 = time.time()
            # torch.cuda.synchronize(end)
            forward_time += (t2 - t1)

            print("forward backward time", t2- t1)
            # print("forward backward time withj timers",start.elapsed_time(end)/1000)
            optimizer.step()
        t02 = time.time()
        total_time_per_epoch.append(t02 - t01)
        if epochs%10 ==0:
            print("Accuracy", correct, total)
        th.join()
        forward_time_per_epoch.append(forward_time)
        graph_slice_time_per_epoch.append(slice_time)
        #cache_load_time_per_epoch.append(sampler.cache_refresh_time)
        # move_batch_time_per_epoch.append(sampler.move_batch_time)
        # extra_stuff_per_epoch.append(sampler.extra_stuff)
        # gpu_slice_per_epoch.append(sampler.gpu_slice)
        # print("Finished one pass !!!")
        # total_loss = torch.reduce(loss(outputs,classes))
        # total_loss.backward()
        # optimizer.step()
    #print("total time per epoch {}".format(total_time_per_epoch))
    #print("batch slice time {}".format(graph_slice_time_per_epoch))
    #print("Forward tie {}".format(forward_time_per_epoch))
    #print("cache load time {}".format(cache_load_time_per_epoc))
    print("avg epoch time {}".format(sum(total_time_per_epoch[1:])/(args.num_epochs - 1)))
    print("avg forward time {}".format(sum(forward_time_per_epoch[1:])/(args.num_epochs - 1)))
    print("batch slice time {}".format(sum(graph_slice_time_per_epoch[1:])/(args.num_epochs -1)))
    #print("cache refresh time {}".format(sum(cache_load_time_per_epoch[1:])/(args.num_epochs - 1)))
    # print("move_batch timer per epoch {}".format(sum(move_batch_time_per_epoch[1:])/(args.num_epochs - 1)))
    # print("extra_stuff_per_epoch {}".format(sum(extra_stuff_per_epoch[1:])/(args.num_epochs - 1)))
    # print("gpu_slice_per_epoch  {}".format(sum(gpu_slice_per_epoch[1:])/(args.num_epochs - 1)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv")
    # training details
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--num-workers', type=int, default=0,
       help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--fsize', type = int, default = -1, help = "use only for synthetic")
    # model name and details
    argparser.add_argument('--debug',type = bool, default = False)
    argparser.add_argument('--cache-per', type =float, default = .25)
    argparser.add_argument('--model-name',help="gcn|gat")
    argparser.add_argument('--num-epochs', type=int, default=5)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads if gat")
    argparser.add_argument('--fan-out', type=str, default='10,10,25')
    argparser.add_argument('--batch-size', type=int, default=(4096))
    argparser.add_argument('--dropout', type=float, default=0)
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    print("")
    train(args)
