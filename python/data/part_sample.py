from data.bipartite import Bipartite
import random
import torch

class Sample:

    def __init__(self, csample):
        self.layers = []
        self.randid = 8
        #random.randint(0,10000)
        self.cache_hit_from= []
        self.cache_hit_to = []
        self.cache_miss_from = []
        self.cache_miss_to = []
        self.out_nodes = []
        for i in range(len(csample.cache_hit_to)):
            self.cache_hit_to.append(csample.cache_hit_to[i])
            self.cache_hit_from.append(csample.cache_hit_from[i])
            self.cache_miss_to.append(csample.cache_miss_to[i])
            self.cache_miss_from.append(csample.cache_miss_from[i])
            self.out_nodes.append(csample.out_nodes[i])

        # print(len(csample.layers))
        for layer in csample.layers:
            l = []
            # print(len(layer))
            for cbipartite in layer:
                bp = Bipartite()
                bp.construct_from_cobject(cbipartite)
                l.append(bp)
            self.layers.append(l)

    def debug(self):
        for ll in self.layers:
            for bp in ll:
                bp.debug()

    def get_number_of_edges(self):
        s = 0
        for l in self.layers:
            for bp in l:
                s += bp.get_number_of_edges()
        return s

class Gpu_Local_Sample:
    # Serialize layer
    def __init__(self):
        self.randid = 0
        self.device_id = 0
        # 3 Layers in grpah
        self.cache_hit_from= torch.tensor([])
        self.cache_hit_to = torch.tensor([])
        self.cache_miss_from =torch.tensor([])
        self.cache_miss_to =torch.tensor([])
        self.out_nodes = torch.tensor([])
        self.debug_val = 0
        self.layers = [Bipartite() for i in range(3)]


    def prepare(self,attention = False):
        last_layer = self.layers[0]
        i = 0
        l = self.layers[0]
        for i in self.layers:
            i.reconstruct_graph(attention)
        self.layers.reverse()

    def debug(self):
        for i in self.layers:
            i.debug()

    def set_from_global_sample(self, global_sample, device_id):
        self.randid = global_sample.randid
        self.layers = []
        for layer in global_sample.layers:
            self.layers.append(layer[device_id])
        # self.last_layer_nodes = global_sample.last_layer_nodes[device_id]
        self.device_id = device_id

        self.cache_hit_from = global_sample.cache_hit_from [device_id]
        self.cache_hit_to = global_sample.cache_hit_to[device_id]
        self.cache_miss_from = global_sample.cache_miss_from[device_id]
        self.cache_miss_to = global_sample.cache_miss_to[device_id]
        self.out_nodes = global_sample.out_nodes[device_id]

    def get_edges_and_send_data(self):
        edges_total = 0
        edges_remote = 0
        edges = []
        nodes = []
        nodes.append(self.layers[0].num_in_nodes_local+ self.layers[0].num_in_nodes_pulled)
        for  graph in self.layers:
            edges_total += graph.indices_L.shape[0] + graph.indices_R.shape[0]
            edges_remote += graph.indptr_R.shape[0] + graph.pull_from_offsets[-1] 
            # nodes_pulled += graph.pull_from_offsets[-1]
            edges.append( graph.indices_L.shape[0] + graph.indices_R.shape[0])
            nodes.append(graph.indptr_L.shape[0] + graph.indptr_R.shape[0])
        return (edges_total, edges_remote, edges, nodes)

