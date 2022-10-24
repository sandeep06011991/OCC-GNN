
import numpy as np
import scipy.sparse
import os
import dgl
import torch


def get_graph_data(dataname):
  """
  Parames:
    dataname: shoud be a folder name, which contains
              adj.npz and feat.npy
  Returns:
    adj, feat, train_mask, val_mask, test_mask, labels
  """

  adj = scipy.sparse.load_npz(
    os.path.join(dataname, 'adj.npz')
  )
  try:
    feat = np.load(
      os.path.join(dataname, 'feat.npy')
    )
  except FileNotFoundError:
    print('random generate feat...')
    import torch
    feat = torch.rand((adj.shape[0], 100))

  return adj, feat


def get_sub_train_graph(dataname, idx, partitions):
  """
  Params:
    dataname: should be a folder name.
              partitions should already be in the 'naive' folder
    idx: sub train partiton id
  Returns:
    adj
    train2fullid
  """
  if "com-orkut" in dataname or  "amazon" in dataname:
      adj_file = os.path.join(dataname, 'adj.npz')
      train2fullid = np.load(os.path.join(dataname, 'vmap.npy'))
      adj = scipy.sparse.load_npz(adj_file)
      # np.range(0,adj_file.shape[0])
      # train2full_file = os.path.join(dataname, 'sub_train2fullid_{}.npy'.format(idx))
  else:
      dataname = os.path.join(dataname, '{}naive'.format(partitions))
      adj_file = os.path.join(dataname, 'subadj_{}.npz'.format(idx))
      train2full_file = os.path.join(dataname, 'sub_train2fullid_{}.npy'.format(idx))
      adj = scipy.sparse.load_npz(adj_file)
      train2fullid = np.load(train2full_file)
  return adj, train2fullid


def get_struct(dataname):
  """
  Params:
    dataname: shoud be a folder name, which contains
              adj.npz in coo matrix format
  """
  adj = scipy.sparse.load_npz(
    os.path.join(dataname, 'adj.npz')
  )
  return adj


def get_masks(dataname):
  """
  Params:
    dataname: shoud be a folder name, which contains
              train_mask, val_mask, test_mask
  """
  train_mask = np.load(
    os.path.join(dataname, 'train.npy')
  )
  val_mask = np.load(
    os.path.join(dataname, 'val.npy')
  )
  test_mask = np.load(
    os.path.join(dataname, 'test.npy')
  )
  return train_mask, val_mask, test_mask


def get_sub_train_nid(dataname, idx, partitions):
  if "com-orkut" in dataname or  "amazon" in dataname:
    "These graphs are too large for pagraphs partitioning algorithm"
    train_nids = os.path.join(dataname, 'train_idx.bin')
    train_nids = np.fromfile(train_nids,dtype = np.int64)
    train_nids = torch.from_numpy(train_nids)
    num_partitions = int(train_nids.shape[0]/partitions)
    train_nids = torch.split(train_nids, num_partitions)[idx].numpy()
    sub_train_nid = train_nids
    # np.range(0,adj_file.shape[0])
    # train2full_file = os.path.join(dataname, 'sub_train2fullid_{}.npy'.format(idx))
  else:
    dataname = os.path.join(dataname, '{}naive'.format(partitions))
    sub_train_file = os.path.join(dataname, 'sub_trainid_{}.npy'.format(idx))
    sub_train_nid = np.load(sub_train_file)
  return sub_train_nid


def get_labels(dataname):
  """
  Params:
    dataname: shoud be a folder name, which contains
              train_mask, val_mask, test_mask
  """
  labels = np.load(
    os.path.join(dataname, 'labels.npy')
  )
  return labels


def get_sub_train_labels(dataname, idx, partitions):
  if "com-orkut" in dataname or "amazon" in dataname:
      labels = get_labels(dataname)
      labels = torch.from_numpy(labels)
      sub_train_id = get_sub_train_nid(dataname, idx, partitions)
      return labels[sub_train_id]
  dataname = os.path.join(dataname, '{}naive'.format(partitions))
  sub_label_file = os.path.join(dataname, 'sub_label_{}.npy'.format(idx))
  sub_label = np.load(sub_label_file)
  return sub_label


def get_feat_from_server(g, nids, embed_name):
  """
  Fetch features of `nids` from remote server in shared CPU
  Params
    g: created from `dgl.contrib.graph_store.create_graph_from_store`
    nids: required node ids
    embed_name: field name, e.g. 'features', 'norm'
  Return:
    feature tensors of these nids (in CPU)
  """
  cpu_frame = g._node_frame[dgl.utils.toindex(nids)]
  return cpu_frame[embed_name]
