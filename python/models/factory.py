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
        self.layers.append(DistSageConv(in_feats, n_hidden, 'sum'))
        for i in range(1, n_layers - 1):
            self.layers.append(DistSageConv(n_hidden, n_hidden, 'sum'))
        self.layers.append(DistSageConv(n_hidden, n_classes, 'sum'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, bipartite_graphs, shuffle_matrices,
                model_owned_nodes, x):
        for l,(layer, bipartite_graph, shuffle_matrix, owned_nodes) in  \
            enumerate(zip(self.layers,bipartite_graphs,shuffle_matrices, \
                            model_owned_nodes)):
            # print("layer attempt ",l)
            x = layer(bipartite_graph, shuffle_matrix, owned_nodes, x)
            # print("layer done ", l)
            if l != len(self.layers)-1:
                x = [self.dropout(self.activation(i)) for i in x]
        return x




def get_model(features, labels):
    # Todo: Add options as inputs and construct the correct model.
    print("Model configuration is hardcoded, read from options instead")
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = 256
    n_class = (torch.max(labels).item()+1)
    n_layers = 3
    activation = torch.nn.ReLU()
    return DistSAGEModel(in_feats, n_hidden, n_class, n_layers , \
        activation, dropout)
