from torch import nn
import dgl.nn.pytorch as dglnn
import dgl
import numpy as np
import torch as th
import torch
import torch.nn as nn

class SAGE(nn.Module):
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
        type = 'mean'
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, type))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, type))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self,blocks,x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


    def inference(self, g, x, device):
        raise Exception("We dont measure inference")
