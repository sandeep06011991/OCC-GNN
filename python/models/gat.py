from torch import nn
import dgl.nn.pytorch as dglnn
import dgl
import numpy as np
import torch as th
import torch
import torch.nn as nn
import time


class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 num_heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # type = 'mean'
        self.layers.append(dglnn.GATConv(in_feats, n_hidden,
                           num_heads, allow_zero_in_degree=True))
        for _ in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(
                n_hidden, n_hidden, num_heads, allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes,
                           num_heads, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = h.flatten(1) if l != len(self.layers) - 1 else h.mean(1)
        return h

    def inference(self, g, x, device):
        raise Exception("We dont measure inference")
