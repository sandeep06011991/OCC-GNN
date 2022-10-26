import os
import sys
import argparse, time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import dgl
from dgl._deprecate.graph import DGLGraph
import os

PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN/upgraded_pagraph"
path_set = False
for p in sys.path:
    # print(p)
    if PATH_DIR ==  p:
       path_set = True
if (not path_set):
    print("Setting Path")
    sys.path.append(PATH_DIR)

FEAT_DICT = {"ogbn-arxiv":128, "ogbn-products":100,\
                "amazon":200, "reorder-papers100M":128, \
                    "com-orkut":400}
N_CLASSES = {"ogbn-arxiv":40, "ogbn-products":48,\
                "amazon":102, "reorder-papers100M":172, \
                    "com-orkut":48}


from PaGraph.model.dgl_sage import  *
from PaGraph.model.dgl_gat import  *

import PaGraph.data as data
import PaGraph.storage as storage
from PaGraph.parallel import SampleLoader
import nvtx

ROOT_DIR = "/work/spolisetty_umass_edu/data/pagraph"


def trainer(rank, args, backend='nccl'):
  dataset = "{}/{}/".format(ROOT_DIR, args.dataset)
  feat_size = FEAT_DICT[args.dataset]
  # init multi process
  # load datai

  rank = 0
  world_size = 4
  dataname = os.path.basename(dataset)
  adj, t2fid = data.get_sub_train_graph(dataset, rank, world_size)

  g = dgl.from_scipy(adj)
  n_classes = N_CLASSES[args.dataset]
  rank = 0
  world_size = 4
  train_nid = data.get_sub_train_nid(dataset, rank, world_size)
  print("Training_nid", train_nid.shape, rank)
  print("Expected number of minibatches",train_nid.shape[0]/args.batch_size)
  emb = train_nid.shape[0]/args.batch_size
  sub_labels = data.get_sub_train_labels(dataset, rank, world_size)
  assert(np.max(train_nid) >100)
  labels = np.zeros(np.max(train_nid) + 1, dtype=np.int)
  labels[train_nid] = sub_labels.flatten()
  train_nid = torch.from_numpy(train_nid)
  # to torch tensor
  t2fid = torch.LongTensor(t2fid)
  labels = torch.LongTensor(labels)
  embed_names = ['features']
  ctx = torch.device(rank)

  sampler = dgl.dataloading.NeighborSampler(
            [int(args.num_neighbors) for i in range(3)], replace = True)
  world_size = 4
  train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
  number_of_minibatches = train_nid.shape[0]/args.batch_size
  print(args.batch_size) 
  print(sampler)
  g = g.add_self_loop()
  print(g)
  print(train_nid.shape)
  dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cpu',
        batch_size=1024,
        shuffle=True,
        drop_last=True,
        num_workers = 2, 
        persistent_workers= True)

  for epoch in range(args.n_epochs):
    it = iter(dataloader)
    while True:
      t00 = time.time()
      try:
          with nvtx.annotate('sample',color = 'yellow'):
             s1 = time.time()
             input_nodes, seeds, blocks = next(it)
             print(input_nodes, seeds, blocks)
             assert(input_nodes.shape[0] == blocks[0].number_of_src_nodes())
             assert(seeds.shape[0] == blocks[-1].number_of_dst_nodes())
             continue
             s2 = time.time()
             epoch_sample_time += (s2 - s1)
      except StopIteration:
          break
        #torch.distributed.barrier()
      #with torch.autograd.profiler.record_function('gpu-load'):
      #if True:
      s0 = time.time()
      e4.record()
      blocks = [b.to(rank) for b in blocks]
      for b in blocks:
         print(b.number_of_dst_nodes(), b, "block")
      s1 = time.time()
      e5.record()
  #if rank == 0:
  #  print(prof.key_averages().table(sort_by='cuda_time_total'))
  print('Total Time: {:.4f}s'.format(toc - tic))
  if rank == 0:
    remote_g.destroy()
  remote_g.proxy = None

  print("All cleaned up")

def train_hook(*args):
  import line_profiler
  from line_profiler import LineProfiler
  prof = LineProfiler()
  train_w = prof(trainer)
  train_w(*args)
  prof.dump_stats('train.lprof')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')

  parser.add_argument("--dataset", type=str, default="None", required = True,
                      help="path to the dataset folder")
  # model arch
  #parser.add_argument("--feat-size", type=int,
  #                    help='input feature size')
  parser.add_argument("--n-epochs", type=int, default=6,
                      help="number of training epochs")
  parser.add_argument("--batch-size", type=int, default=1032,
                      help="batch size")
  parser.add_argument("--weight-decay", type=float, default=0,
                      help="Weight for L2 loss")
  # sampling hyper-params
  parser.add_argument("--num-neighbors", type=int, default=20,
                      help="number of neighbors to be sampled")
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--remote-sample", dest='remote_sample', action='store_true')
  parser.set_defaults(remote_sample=False)

  args = parser.parse_args()
  rank = 0
  trainer(rank, args)

import os
import sys
import argparse, time
import torch
