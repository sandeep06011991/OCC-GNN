    # Creates models  by parsing the model parametersself.
import torch
from torch import nn
from layers.dist_sageconv import DistSageConv
from layers.dist_gatconv import DistGATConv
from layers.pull import *
# Move this function to seperate file after first forward and back pass
class DistGATModel(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 gpu_id, is_pulled, num_gpus,
                 queues = None, deterministic = False, skip_shuffle = False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.queues = queues
        self.num_heads = 4
        self.num_gpus = num_gpus
        self.gpu_id = gpu_id
        num_heads = self.num_heads
        # allows us to use the hidden dimension for both GCN and GAT
        n_hidden = (int)(n_hidden / num_heads)
        self.layer_events = [torch.cuda.Event(enable_timing = 1)]
        self.layers.append(DistGATConv(in_feats, n_hidden, gpu_id, num_gpus,
                    deterministic = deterministic, num_heads = num_heads, skip_shuffle = skip_shuffle))
        for i in range(1, n_layers - 1):
            self.layers.append(DistGATConv(n_hidden * num_heads, n_hidden,  gpu_id,  num_gpus, num_heads = num_heads, deterministic = deterministic, skip_shuffle = skip_shuffle))
            self.layer_events.append(torch.cuda.Event(enable_timing = 1))
        self.layers.append(DistGATConv(n_hidden * num_heads, n_classes, gpu_id,  num_gpus, num_heads = 1, deterministic = deterministic, skip_shuffle = skip_shuffle))
        self.layer_events.append(torch.cuda.Event(enable_timing = 1))
        self.layer_events.append(torch.cuda.Event(enable_timing = 1))
        assert(len(self.layers) + 1 == len(self.layer_events))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_pulled = is_pulled
        self.deterministic = deterministic
        self.fp_end = torch.cuda.Event(enable_timing=True)
        self.bp_end = torch.cuda.Event(enable_timing=True)

    def forward(self, bipartite_graphs, x, testing = False):
        t = []
        for l,(layer, bipartite_graph) in  \
            enumerate(zip(self.layers,bipartite_graphs.layers)):
            t1 = time.time()
            if(self.is_pulled):
                pass
            self.layer_events[l].record()
            x = pull(bipartite_graph, x, self.gpu_id, self.num_gpus,  l)
            x = layer(bipartite_graph, x,l, testing )
            t2 = time.time()
            if l != len(self.layers)-1:
                x = self.dropout(self.activation(x))
            #t2 = time.time()
            t.append(t2-t1)
        if self.gpu_id == 0:
            print(t)
        self.layer_events[len(self.layers)].record()    
        return x


    def print_grad(self):
        for id,l in enumerate(self.layers):
            l.print_grad(id)

    def get_reset_shuffle_time(self):
        s = 0
        for l in self.layers:
            s += l.get_reset_shuffle_time()
        return s

    def get_reset_layer_time(self):
       events = self.layer_events 
       events[-1].synchronize()
       first_layer = events[0].elapsed_time(events[1])/1000
       other_layer = events[1].elapsed_time(events[-1])/1000
       return first_layer, other_layer      

def get_gat_distributed(hidden, features, num_classes, gpu_id, deterministic, model, is_pulled, num_gpus, n_layers, skip_shuffle):
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = hidden
    n_classes = num_classes
    if deterministic:
        n_hidden = 1
        n_classes = 1

    activation = torch.nn.ReLU()
    assert(model==  "gat" or model == "gat-pull")
    return DistGATModel(in_feats, n_hidden, n_classes, n_layers, activation, \
            dropout, gpu_id, is_pulled,  num_gpus, deterministic = deterministic , skip_shuffle = skip_shuffle)
