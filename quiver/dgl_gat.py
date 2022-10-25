import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch as th
import dgl

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, \
                out_channels, heads, num_layers, activation, dropout):
        super(GAT,self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(dglnn.GATConv(in_channels, hidden_channels, num_heads = heads))
        for _ in range(num_layers -2):
            self.convs.append(dglnn.GATConv(hidden_channels *heads, hidden_channels,num_heads = heads))
        self.convs.append(dglnn.GATConv(hidden_channels * heads, out_channels, num_heads = 1))
        self.n_classes = out_channels
        self.n_heads = heads
        self.n_hidden = hidden_channels
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.convs) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h = h.flatten(1)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        self.layers = self.convs
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden * self.n_heads if l != len(
                self.layers) - 1 else self.n_classes).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                device='cpu',
                batch_size=1032,
                shuffle=False,
                drop_last=False,
                num_workers=2)

            for input_nodes, output_nodes, blocks in (dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes].to(device)
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                h = h.flatten(1)
                y[output_nodes] = h

            x = y
        return y
