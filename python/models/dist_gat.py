    # Creates models  by parsing the model parametersself.
import torch
from torch import nn
from layers.dist_sageconv import DistSageConv
from layers.dist_gatconv import DistGATConv
# Move this function to seperate file after first forward and back pass
class DistGATModel(torch.nn.Module):

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
        self.num_heads = 3
        num_heads = self.num_heads
        self.layers.append(DistGATConv(in_feats, n_hidden, gpu_id,
                    deterministic = deterministic, num_heads = num_heads))
        for i in range(1, n_layers - 1):
            self.layers.append(DistGATConv(n_hidden * num_heads, n_hidden,  gpu_id, num_heads = 3, deterministic = deterministic))
        self.layers.append(DistGATConv(n_hidden * num_heads, n_classes, gpu_id,num_heads = 1, deterministic = deterministic))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.deterministic = deterministic
        self.fp_end = torch.cuda.Event(enable_timing=True)
        self.bp_end = torch.cuda.Event(enable_timing=True)

    def forward(self, bipartite_graphs, x, in_degrees, testing = False):

        for l,(layer, bipartite_graph) in  \
            enumerate(zip(self.layers,bipartite_graphs.layers)):
            x = layer(bipartite_graph, x,l, in_degrees, testing )
            if l != len(self.layers)-1:
                x = self.dropout(self.activation(x))
        return x


    def print_grad(self):
        for id,l in enumerate(self.layers):
            l.print_grad(id)



def get_gat_distributed(hidden, features, num_classes, gpu_id, deterministic, model):
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = hidden
    n_classes = num_classes
    if deterministic:
        n_hidden = 1
        n_classes = 1

    n_layers = 3
    activation = torch.nn.ReLU()
    assert(model==  "gat")
    return DistGATModel(in_feats, n_hidden, n_classes, n_layers, activation, \
            dropout, gpu_id, deterministic = deterministic )
