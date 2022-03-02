# Creates models  by parsing the model parametersself.
import torch
from torch import nn
from layers.dist_sageconv import DistSageConv

# Move this function to seperate file after first forward and back pass
class DistSAGEModel(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(DistSageConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(DistSageConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(DistSageConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, bipartite_graphs, x):
        for l,(layer, bipartite_graph, shuffle_matrix) in  \
            enumerate(zip(self.layers,bipartite_graphs,shuffle_matrices)):
            x = layer(bipartite_graph, shuffle_matrix, x)
            if l != len(self.layers)-1:
                x = [self.dropput(self.activation(i)) for i in x]
        return x

    


def get_model():
    # Todo: Add options as inputs and construct the correct model.
    dropout = .1
    in_feats = 1024
    n_hidden = 128
    n_class = 40
    n_layers = 3
    activation = torch.nn.ReLU()
    return DistSAGEModel(in_feats, n_hidden, n_class, n_layers , \
        activation, dropout)
