from data.bipartite import Bipartite
import random
import torch

class Sample:

    def __init__(self, csample):
        self.in_nodes = csample.in_nodes
        self.out_nodes = csample.out_nodes
        self.layers = []

        self.randid = random.randint(0,10000)
        self.missing_node_ids = []
        self.debug_vals  = []
        self.cached_node_ids = []
        for i in range(4):
            self.missing_node_ids.append(csample.missing_node_ids[i])
            self.debug_vals.append(csample.debug_vals[i])
            self.cached_node_ids.append(csample.cached_node_ids[i])

        # print(len(csample.layers))
        for layer in csample.layers:
            l = []
            # print(len(layer))
            for cbipartite in layer:
                bp = Bipartite()
                bp.construct_from_cobject(cbipartite)
                l.append(bp)
            self.layers.append(l)

    def get_number_of_edges(self):
        s = 0
        for l in self.layers:
            for bp in l:
                s += bp.get_number_of_edges()
        return s

class Gpu_Local_Sample:
    # Serialize layer
    def __init__(self):
        self.in_nodes = 0
        self.out_nodes = 0
        self.randid = 0
        self.device_id = 0
        # 3 Layers in grpah
        self.missing_node_ids = torch.tensor([])
        self.cached_node_ids = torch.tensor([])
        self.debug_val = 0
        self.layers = [Bipartite() for i in range(3)]

    def prepare(self):
        self.last_layer_nodes = []
        last_layer = self.layers[0]
        i = 0
        l = self.layers[0]
        # FixME: Make everything into longs
        self.last_layer_nodes = l.out_nodes[l.owned_out_nodes]
        for i in self.layers:
            i.reconstruct_graph()
        self.layers.reverse()

    def set_from_global_sample(self, global_sample, device_id):
        self.in_nodes = global_sample.in_nodes
        self.out_nodes = global_sample.out_nodes
        self.randid = global_sample.randid
        self.layers = []
        for layer in global_sample.layers:
            self.layers.append(layer[device_id])
        # self.last_layer_nodes = global_sample.last_layer_nodes[device_id]
        self.device_id = device_id
        self.missing_node_ids = global_sample.missing_node_ids[device_id]
        self.cached_node_ids = global_sample.cached_node_ids[device_id]
        self.debug_val = global_sample.debug_vals[device_id]

    def get_edges_and_send_data(self):
        edges = 0
        nodes_moved = 0
        for  graph in self.layers:
            edges += graph.indices.shape[0]
            for j in range(4):
                nodes_moved += graph.to_ids[j].shape[0]
        return (edges,nodes_moved)
