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


    def forward(self, bipartite_graphs, x):
        for l,(layer, bipartite_graph) in  \
            enumerate(zip(self.layers,bipartite_graphs)):
            print("layer attempt ",l)
            import time
            t1 = time.time()
            x = layer(bipartite_graph, x)
            t2 = time.time()
            # print("layer time",t2-t1)
            # print("layer done ", l)
            if l != len(self.layers)-1:
                x = [self.dropout(self.activation(i)) for i in x]
            print("first layer done !!!")    
        return x



def get_model(hidden, features, num_classes):
    # Todo: Add options as inputs and construct the correct model.
    print("Model configuration is hardcoded, read from options instead")
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = hidden
    n_class = num_classes
    n_layers = 3
    activation = torch.nn.ReLU()
    return DistSAGEModel(in_feats, n_hidden, n_class, n_layers , \
        activation, dropout)
