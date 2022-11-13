
from utils import get_process_graph
import dgl
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.multiprocessing import Queue

from sklearn.linear_model import LinearRegression

def sample_statistics(graph, fanout, no_layers, batch_size, num_workers = 0):
    graph, partition_map, num_classes = get_process_graph(graph)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for _ in range(no_layers)], replace = True)
    train_nid = graph.ndata['train_mask'].nonzero()
    train_nid = train_nid.flatten()
    dataloader_i = dgl.dataloading.NodeDataLoader(
        graph,
        train_nid,
        sampler,
        device='cpu',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)
    avg_epoch = []
    avg_sample = []
    avg_fanout = []
    avg_edges = []
    avg_degree = []
    serialization_cost = []
    avg  = []
    q = Queue()
    for i in range(10):
        dataloader = iter(dataloader_i)
        t11 = time.time()
        while(True):
            try:
                t1 = time.time()
                input_nodes, seeds, blocks = next(dataloader)
                t2 = time.time()
                avg.append(t2-t1)
                time.sleep(t2-t1)
                q.put((input_nodes,seeds,blocks))
                q.get()
                t3 = time.time()
                avg_edges.append(blocks[0].number_of_edges())
                avg_fanout.append(input_nodes.shape[0]/seeds.shape[0])
                serialization_cost.append(t3-t2)
            except StopIteration:
                break
        t22 = time.time()
        assert(statistics.variance(avg[1:])/statistics.mean(avg[1:]) < .1)
        avg_sample.append(sum(avg[1:])/len(avg[1:]))
        avg_epoch.append(t22-t11)
    # print("sample", avg)
    assert(statistics.variance(avg_fanout[1:])/statistics.mean(avg_fanout[1:])<.1)
    avg_fanout = (sum(avg_fanout)/len(avg_fanout))
    avg_edges = (sum(avg_edges))/len(avg_edges)
    # print("avg fanout", avg_fanout)
    # print(statistics.variance(avg_epoch[1:]) / statistics.mean(avg_epoch[1:]))
    print(avg_epoch)
    print(statistics.variance(avg_epoch[1:]) / statistics.mean(avg_epoch[1:]))
    assert(statistics.variance(avg_epoch[1:]) / statistics.mean(avg_epoch[1:]) < .1)
    assert(statistics.variance(avg_sample[1:])/ statistics.mean(avg_epoch[1:]) < .10)
    minibatch_sample = sum(avg_sample[1:])/len(avg_sample[1:])
    epoch_sample = sum(avg_epoch[1:])/len(avg_epoch[1:])
    assert(statistics.variance(serialization_cost) / statistics.mean(serialization_cost) < .1)
    return {"minibatch_sample": minibatch_sample, "epoch_sample":epoch_sample, \
            "avg_fanout": avg_fanout, "avg_edges":avg_edges, "serialization": statistics.mean(serialization_cost)}

'''
Cost model of sampler. Sampler at the following cost.
Given n and a fanout seeds, Cost of sampling one layer is n * f
reordering them if using a map n * f or it is n(log(n))
a (n * f) + b(n(log(n))) + c
'''
def single_layer_sampler(graphs):
    fanout = 20
    f = fanout
    no_layers = 1
    fig = plt.figure()
    colors = ["red", "green", "yellow"]
    i = 0
    X = []
    Y = []
    xb = [256, 256*4, 4096, 4096 * 4]
    for graph in graphs:
        y = []
        for batch_size in xb:
            n = batch_size

            X.append(expected)
            run_time = sample_statistics(graph, fanout, no_layers, batch_size)
            values = run_time["minibatch_sample"]
            f = run_time["avg_fanout"]
            expected = [n * f, n * f * math.log(n * f)]
            y.append(values)
            Y.append(values)
        plt.scatter(xb, y, color = colors[i])
        i = i + 1

    reg = LinearRegression().fit(np.array(X), np.array(Y))
    x_in = [[i * f, (i * f) * math.log(i * f)] for i in xb]
    plt.plot(xb, reg.predict(np.array(x_in)).flatten().tolist())
    fig.savefig("check.png")

def batch_size_model(graphs):
    for graph in graphs:
        fanout = 2
        batch_size = 4096
        no_layers = 1
        for workers in [0,2,4,8,16]:
            run_time = sample_statistics(graph, fanout, no_layers, batch_size, workers)
            print(workers, run_time)
    pass

# (Cost of geerating + Cost of Serializing ) = 1 Worker
# (Cost of generting / n + Cost of serializing)
# (Cost of serializing queues. )
def multiprocessing_sampler_benefits():
    graph = "ogbn-arxiv"
    fanout = 10
    no_layers = 1
    batch_size = 4096
    for workers in [0,1]:
        run_time = sample_statistics(graph, fanout, no_layers, batch_size, workers)
        print(workers, run_time)

def cost_of_serialization():
    queue = torch.Queue()
    data_put = {}
    M = 1024 * 256
    G = 1024 * 1024 * 256
    runs = 4
    data_size = [10 * M , 50 * M, 100 * M, 500 * M, 1 * G, 3 * G]
    for ds in data_size:
        data = torch.rand(ds,)
        avg_put = []
        avg_get = []
        for _ in range(runs):
            t1 = time.time()
            queue.put(data)
            t2 = time.time()
            queue.get(data)
            t3 = time.time()
            avg_put.append(t2-t1)
            avg_get.append(t3-t2)
        # data_put.

if __name__ == "__main__":
    name = ["reorder-papers100M","ogbn-products","amazon"]
    name = ["ogbn-arxiv","ogbn-products","reorder-papers100M"]
    # batch_size_model()
    multiprocessing_sampler_benefits()
