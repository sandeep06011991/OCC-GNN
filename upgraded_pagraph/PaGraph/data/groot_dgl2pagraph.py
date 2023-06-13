"""
Change built in dgl graph dataset into the format of PaGraph
"""

import os
import argparse
import numpy as np
import scipy.sparse
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import os
import time 
import dgl 

from os.path import exists
ROOT_DIR = "/home/ubuntu/data"
TARGET_DIR = ROOT_DIR + "/pagraph"

def convert_ogb_data(dataset_name):
  t1 = time.time()
  dataset = DglNodePropPredDataset(dataset_name, root=ROOT_DIR)
  graph, labels = dataset[0]
  edges = graph.edges()
  graph.remove_edges(torch.where(edges[0] == edges[1])[0])
  edges = graph.edges()
  t2 = time.time()
  print(f"Time to read graph:{t2-t1}")
  num_edges = graph.num_edges()
  num_nodes = graph.num_nodes()
  assert(num_edges == edges[0].shape[0])
  features = graph.ndata['feat']
  assert features.shape[0] == num_nodes
  splits = dataset.get_idx_split()
  out_folder = TARGET_DIR + "/" + dataset_name
  os.makedirs(out_folder, exist_ok = True)

  def get_mask(nodes, num_nodes):
    mask = torch.zeros(num_nodes, dtype = torch.int64)
    mask[nodes] = 1
    return mask
  dgl.data.save_graphs(out_folder + "/adj.npz", graph)
  np.save(os.path.join(out_folder, 'feat.npy'), features.numpy())
  np.save(os.path.join(out_folder, 'labels.npy'), labels.numpy())
  np.save(os.path.join(out_folder, 'train.npy'), get_mask(splits["train"], num_nodes).numpy())
  np.save(os.path.join(out_folder, 'val.npy'), get_mask(splits["valid"], num_nodes).numpy())
  np.save(os.path.join(out_folder, 'test.npy'),get_mask(splits["test"], num_nodes).numpy())
  print('Convert Finishes')  

def convert_reddit_data(dataset, out_folder, self_loop=False):
  """
  Load DGL graph dataset
  """
  self_loop_str = ""
  if self_loop:
    self_loop_str = "_self_loop"
  download_dir = get_download_dir()
  extract_dir = os.path.join(download_dir, "{}{}".format(dataset, self_loop_str))

  coo_adj = scipy.sparse.load_npz(os.path.join(extract_dir, "{}{}_graph.npz"
                                     .format(dataset, self_loop_str)))

  reddit_data = np.load(os.path.join(extract_dir, "{}_data.npz".format(dataset)))
  features = reddit_data["feature"]
  labels = reddit_data["label"]
  node_types = reddit_data["node_types"]
  train_mask = (node_types == 1)
  val_mask = (node_types == 2)
  test_mask = (node_types == 3)

  scipy.sparse.save_npz(os.path.join(out_folder, 'adj.npz'), coo_adj)
  np.save(os.path.join(out_folder, 'feat.npy'), features)
  np.save(os.path.join(out_folder, 'labels.npy'), labels)
  np.save(os.path.join(out_folder, 'train.npy'), train_mask)
  np.save(os.path.join(out_folder, 'val.npy'), val_mask)
  np.save(os.path.join(out_folder, 'test.npy'), test_mask)

  print('Convert Finishes')


if __name__ == '__main__':
  dataset = "ogbn-arxiv"
  convert_ogb_data(dataset)
  