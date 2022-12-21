
from utils.utils import get_process_graph
from cslicer import cslicer
from data.part_sample import *
from utils.utils import *


def get_average_skew(graph_name, batch_size):
    graph, _ , _ = get_process_graph(graph_name, -1)
    fanout = 20
    deterministic = False
    self_edge = False
    rounds = 4
    pull_optimization = False
    storage_vector = [[],[],[],[]]
    testing = False
    num_layers = 3
    sampler = cslicer(graph_name,storage_vector,fanout, deterministic, testing , self_edge, rounds, \
            pull_optimization, num_layers)
    
    training_nodes = graph.ndata['train_mask'].nonzero().flatten()
    training_nodes = training_nodes.tolist()

    i = 0
    avg_skew = []
    remote_edge_ratio = []
    for _ in range(5):
        i = 0
        random.shuffle(training_nodes)
        while (i < len(training_nodes)):
            csample = sampler.getSample(training_nodes[i:i+batch_size])
            csample = Sample(csample)
            edges_per_gpu = []
            total_edges = 0
            remote_edges = 0
            print(i, len(training_nodes))
            for j in range(4):

                gs = Gpu_Local_Sample()
                gs.set_from_global_sample(csample, j)
                te, re = gs.get_edges_and_send_data()
                total_edges += te
                remote_edges += re
                edges_per_gpu.append(total_edges)
            remote_edge_ratio.append(remote_edges/total_edges)
            avg_skew.append(((max(edges_per_gpu) - min(edges_per_gpu)) * 4)/sum(edges_per_gpu))
            i = i + batch_size
    avg_skew = sum(avg_skew)/len(avg_skew)    
    avg_remote = sum(remote_edge_ratio)/len(remote_edge_ratio)
    with open("{}/microbench/skew_exp.txt".format(OUT_DIR),'a') as fp:
        fp.write("{}|{}|{}|{}\n".format(graph_name, batch_size, avg_skew, avg_remote))




if __name__ == "__main__":
    graph_name = "ogbn-arxiv"
    batch_sizes = [ 4096, 4096 * 4]
    batch_sizes = [1024]
    for graph_name in [ "reorder-papers100M", "amazon"]:
    #for graph_name in ["ogbn-arxiv"]:
        for batch_size in batch_sizes:
            get_average_skew(graph_name, batch_size)

