import importlib
import os.path as osp

import torch

from torch_geometric_autoscale import (get_data, metis, permute,
                                       SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_micro_f1, dropout)
# from torch_geometric_autoscale.data import get_products
ROOT_DIR = '/home/spolisetty/datadir'
data,in_channels,out_channels = get_data(root = ROOT_DIR,name = 'arxiv')

perm, ptr = metis(data.adj_t, num_parts=80, log=True)
data = permute(data, perm, log=True)

train_loader = SubgraphLoader(data, ptr, batch_size=40,
                              shuffle=True, num_workers=2,
                              persistent_workers=True)

for i, (batch, batch_size, *args) in enumerate(train_loader):
    x = batch.x.to(model.device)
    adj_t = batch.adj_t.to(model.device)
    y = batch.y[:batch_size].to(model.device)
    train_mask = batch.train_mask[:batch_size].to(model.device)
