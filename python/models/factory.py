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
        self.deterministic = deterministic
        self.fp_end = torch.cuda.Event(enable_timing=True)
        self.bp_end = torch.cuda.Event(enable_timing=True)
        
    def forward(self, bipartite_graphs, x, in_degrees):
        #print("input ",x.shape,x.device)
        for l,(layer, bipartite_graph) in  \
            enumerate(zip(self.layers,bipartite_graphs.layers)):
            import time
            t1 = time.time()
            # assert(not torch.any(torch.sum(x,1)==0))
            if self.deterministic:
                print("layer edges",l,bipartite_graph.graph.num_edges())
                if(torch.any(torch.sum(x,1)==0)):
                    y = torch.where(torch.sum(x,1)==0)[0][0]
                    print("features size", x.shape)
                    print("is grad", features.requires_grad)
                    print("Found zero in feats at ",l,"local_id",y)
            self.fp_end.record()
            x = layer(bipartite_graph, x,l, in_degrees)
            self.bp_end.record()
            torch.cuda.synchronize(self.bp_end)
            t2 = time.time()
            if l != len(self.layers)-1:
                x = self.dropout(self.activation(x))

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
    # activation = torch.nn.LeakyReLU()
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
