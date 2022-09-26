from torch import nn
import dgl.nn.pytorch as dglnn
import dgl
import numpy as np
import torch as th
import torch
import torch.nn as nn
import time
import dgl.function as fn
class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 deterministic):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        type = 'mean'
        if n_layers == 1:
            self.layers.append(dglnn.SAGEConv(
                in_feats, n_classes, type, bias=False))
        else:
            self.layers.append(dglnn.SAGEConv(
                in_feats, n_hidden, type, bias=False))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(
                    n_hidden, n_hidden, type, bias=False))
            self.layers.append(dglnn.SAGEConv(
                n_hidden, n_classes, type, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        if deterministic:
            self.set_ones()

    def set_ones(self):
        for l in self.layers:
            l.fc_self.weight = torch.nn.Parameter(
                torch.ones(l.fc_self.weight.shape))
            l.fc_neigh.weight = torch.nn.Parameter(
                torch.ones(l.fc_neigh.weight.shape))
                
    def print_gradient(self):
        for id,l in enumerate(self.layers):
            print("layer self ",id,l.fc_self.weight[:3,0], torch.sum(l.fc_self.weight))
            print("layer neigh",id,l.fc_neigh.weight[:3,0], torch.sum(l.fc_neigh.weight))
            if l.fc_self.weight.grad != None:
                print("layer self grad",id,l.fc_self.weight.grad[:3,0], torch.sum(l.fc_self.weight.grad))
                print("layer neigh grad",id,l.fc_neigh.weight.grad[:3,0], torch.sum(l.fc_neigh.weight.grad))
                if id ==2:
                    print("layer self grad", id, l.fc_self.weight.grad)
                    print("layer neigh grad", id, l.fc_neigh.weight.grad)

                # print("layer neigh",l.fc_neigh.weight[:3,0],torch.sum(l.fc_neigh.weight), torch.sum(l.fc_neigh.weight.grad))

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
