    # Creates models  by parsing the model parametersself.
import torch
from torch import nn
from layers.multiprocess_dist_sageconv import DistSageConv

# Move this function to seperate file after first forward and back pass
class DistSAGEModel(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 gpu_id,
                 queues = None, deterministic = False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.queues = queues
        self.layers.append(DistSageConv(in_feats, n_hidden, gpu_id, aggregator_type = 'sum', queues = queues, deterministic = deterministic))
        for i in range(1, n_layers - 1):
            self.layers.append(DistSageConv(n_hidden, n_hidden,  gpu_id, aggregator_type = 'sum', queues = queues, deterministic = deterministic))
        self.layers.append(DistSageConv(n_hidden, n_classes, gpu_id, aggregator_type = 'sum', queues = queues, deterministic = deterministic))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation


    def forward(self, bipartite_graphs, x, features):
        for l,(layer, bipartite_graph) in  \
            enumerate(zip(self.layers,bipartite_graphs.layers)):
            import time
            t1 = time.time()
            if l == 0:
            #     print("checking")
                indices = bipartite_graph.graph.adj_sparse('csc')[1]
                self_ids_in = bipartite_graph.self_ids_in
                # print("self ids in", self_ids_in)
                in_nodes = torch.cat([indices,self_ids_in])
                a = x[torch.unique(in_nodes)]
                print("layer 0 input sum",torch.sum(a))
            #     # print(features.shape)
            #     d = torch.sort(bipartite_graph.in_nodes)[0]
            #     # print(d)
            #     b = features[torch.sort(bipartite_graph.in_nodes)[0]]
            #     # print(a.shape,b.shape)
            #     if not torch.all(a==b):
            #         print("in_nodes",in_nodes)
            #         print("self_ids_in")
            #         print("Goof up", a ,b)
            #     assert(torch.all(a == b))
            if(l == 0):
                if (torch.any(torch.sum(x,1))== 0):
                    print("input features corrupted")
                    print("gpu local id", torch.where(torch.sum(x,1))[0])
                assert(not torch.any(torch.sum(x,1)==0))
            print("layer edges",l,bipartite_graph.graph.num_edges())
            x = layer(bipartite_graph, x,l)
            assert(not torch.any(torch.sum(x,1)==0))
            t2 = time.time()
            if l != len(self.layers)-1:
                # x = self.dropout(x)
                x = self.dropout(self.activation(x))
            print("layer sum",l, torch.sum(x))
        return x
    def print_grad(self):
        for id,l in enumerate(self.layers):
            l.print_grad(id)


def get_model(hidden, features, num_classes):
    # Todo: Add options as inputs and construct the correct model.
    print("Model configuration is hardcoded, read from options instead")
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = hidden
    n_class = num_classes
    # Fix me: Remove this later.
    n_layers = 3
    activation = torch.nn.ReLU()
    return DistSAGEModel(in_feats, n_hidden, n_class, n_layers , \
        activation, dropout, queues = queues)

def get_model_distributed(hidden, features, num_classes, gpu_id, deterministic):
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = hidden
    n_classes = num_classes
    n_layers = 3
    activation = torch.nn.ReLU()
    return DistSAGEModel(in_feats, n_hidden, n_classes, n_layers, activation, \
            dropout, gpu_id, deterministic = deterministic )
