    # Creates models  by parsing the model parametersself.
import torch
from torch import nn
from layers.dist_sageconv import DistSageConv
# from layers.dist_gatconv import DistGATConv
# Move this function to seperate file after first forward and back pass
class DistSAGEModel(torch.nn.Module):

    def _backward_hook(self, module, grad_input, grad_output):
        print("Backward hook runs !!!!!!!!!!!!!")
        print("grad input", grad_input.shape)
        # for i in grad_out:
        #     print("grad output", i)

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 gpu_id, num_gpus,
                 queues = None, deterministic = False, skip_shuffle = False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.queues = queues
        self.layers.append(DistSageConv(in_feats, n_hidden, gpu_id,  num_gpus, deterministic = deterministic, skip_shuffle = skip_shuffle))
        for i in range(1, n_layers - 1):
            self.layers.append(DistSageConv(n_hidden, n_hidden,  gpu_id,  num_gpus, deterministic = deterministic, skip_shuffle = skip_shuffle))
        self.layers.append(DistSageConv(n_hidden, n_classes, gpu_id,  num_gpus, deterministic = deterministic, skip_shuffle = skip_shuffle))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.deterministic = deterministic
        self.num_gpus = num_gpus
        self.fp_end = torch.cuda.Event(enable_timing=True)
        self.bp_end = torch.cuda.Event(enable_timing=True)

    def get_reset_shuffle_time(self):
        s = 0
        for i in self.layers:
            s += i.get_reset_shuffle_time()
        return s

    def forward(self, bipartite_graphs, in_features, testing = False):
        x = in_features
        for l,(layer, bipartite_graph) in  \
            enumerate(zip(self.layers,bipartite_graphs.layers)):
            import time
            t1 = time.time()
            # assert(not torch.any(torch.sum(x,1)==0))
            # self.fp_end.record()
            # print("in layer ", l)
            x = layer(bipartite_graph, x, l)
            # print("layer done", l)
            # self.bp_end.record()
            # torch.cuda.synchronize(self.bp_end)
            t2 = time.time()
            if l != len(self.layers)-1:
                x = self.dropout(self.activation(x))
        return x


    def print_grad(self):
        for id,l in enumerate(self.layers):
            l.print_grad(id)



def get_sage_distributed(hidden, features, num_classes, gpu_id, deterministic, model, num_gpus, n_layers, skip_shuffle):
    dropout = 0
    in_feats = features.shape[1]
    n_hidden = hidden
    n_classes = num_classes
    if deterministic:
        n_hidden = 1
        n_classes = 1

    activation = torch.nn.ReLU()
    assert(model==  "gcn")
    return DistSAGEModel(in_feats, n_hidden, n_classes, n_layers, activation, \
            dropout, gpu_id, num_gpus, deterministic = deterministic, skip_shuffle = skip_shuffle )
