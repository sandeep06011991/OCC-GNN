import dgl
import torch
import dgl.function as fn
from convert_dgl_dataset import *

TARGET_DIR = "/mnt/homes/spolisetty/data/tests/gcn"
name = "pubmed"
target = TARGET_DIR +"/pubmed"
import os
os.makedirs(target,exist_ok = True)
dataset = get_dataset(name)
write_dataset_dataset(dataset, target)

g = dataset[0]

features = g.ndata['feat']
features.requires_grad = True
labels = g.ndata['label']
g.update_all(fn.copy_u('feat', 'm'), fn.sum('m', 'h_sum'))
g.ndata['h_sum'].sum().backward()

with open(TARGET_DIR+'/aggr.bin','wb') as fp:
    fp.write(g.ndata['h_sum'].detach().numpy().astype('float32').tobytes())
with open(TARGET_DIR+'/grad.bin','wb') as fp:
    fp.write(g.ndata['feat'].grad.numpy().astype('float32').tobytes())

print("All data written !!")
