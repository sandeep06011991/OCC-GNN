from data.bipartite import Bipartite

class Sample:

    def __init__(self, csample):
        self.in_nodes = csample.in_nodes
        self.out_nodes = csample.out_nodes
        self.layers = []

        self.randid = random.randint(0,10000)
        # print(len(csample.layers))
        for layer in csample.layers:
            l = []
            # print(len(layer))
            for cbipartite in layer:
                l.append(Bipartite(cbipartite))
            self.layers.append(l)


class Gpu_Local_Sample:
    # Serialize layer
    def __init__(self):
        self.in_nodes = 0
        self.out_nodes = 0
        self.randid = 0
        # 3 Layers in grpah
        self.layers = [Bipartite() for i in range(3)]

    def set_from_global_sample(self, global_sample):
        self.in_nodes = global_sample.in_nodes
        self.out_nodes = global_sample.out_nodes
        self.randid = global_sample.randid
        self.layers = []
        for layer in global_sample.layers:
            self.layers.append(layer[device_id])
        # self.last_layer_nodes = global_sample.last_layer_nodes[device_id]
        self.device_id = device_id
