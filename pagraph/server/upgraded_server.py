import sys
import os
import argparse
import numpy as np
import torch
import dgl
from dgl._deprecate.graph import DGLGraph
from dgl.contrib.sampling import SamplerPool
import dgl.function as fn
import multiprocessing


ROOT_DIR = "/work/spolisetty_umass_edu/data/pagraph"

PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN/pagraph"
path_set = False
for p in sys.path:
    print(p)
    if PATH_DIR ==  p:
       path_set = True
if (not path_set):
    print("Setting Path")
    sys.path.append(PATH_DIR)

import PaGraph.data as data
from PaGraph.parallel import SampleDeliver


def main(args):
  print("Start server")
  
  dataset = "{}/{}/".format(ROOT_DIR, args.dataset)
  coo_adj, feat = data.get_graph_data(dataset)

  graph = DGLGraph(coo_adj, readonly=True)
  features = torch.FloatTensor(feat)

  graph_name = os.path.basename(dataset)
  vnum = graph.number_of_nodes()
  enum = graph.number_of_edges()
  feat_size = feat.shape[1]

  print('=' * 30)
  print("Graph Name: {}\nNodes Num: {}\tEdges Num: {}\nFeature Size: {}"
        .format(graph_name, vnum, enum, feat_size)
  )
  print('=' * 30)
  sys.stdout.flush()
  # create server
  g = dgl.contrib.graph_store.create_graph_store_server(
        graph, graph_name,
        True, args.num_workers, 
        80)
  
  # calculate norm for gcn
  dgl_g = DGLGraph(graph, readonly=True)

  if args.model == 'gcn':
    dgl_g = DGLGraph(graph, readonly=True)
    norm = 1. / dgl_g.in_degrees().float().unsqueeze(1)
    # preprocess 
    if args.preprocess:
      print('Preprocessing features...')
      dgl_g.ndata['norm'] = norm
      dgl_g.ndata['features'] = features
      dgl_g.update_all(fn.copy_src(src='features', out='m'),
                       fn.sum(msg='m', out='preprocess'),
                       lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
      features = dgl_g.ndata['preprocess']
    g.ndata['norm'] = norm
    g.ndata['features'] = features
    del dgl_g

  elif args.model == 'graphsage':
    if args.preprocess: # for simple preprocessing
      print('preprocessing: warning: jusy copy')
      g.ndata['neigh'] = features
    g.ndata['features'] = features

  # remote sampler 
  if args.sample:
    subgraph = []
    sub_trainnid = []
    for rank in range(args.num_workers):
      subadj, _ = data.get_sub_train_graph(dataset, rank, args.num_workers)
      train_nid = data.get_sub_train_nid(dataset, rank, args.num_workers)
      subgraph.append(dgl.DGLGraph(subadj, readonly=True))
      sub_trainnid.append(train_nid)
    hops = args.gnn_layers - 1 if args.preprocess else args.gnn_layers
    print('Expected trainer#: {}. Start sampling at server end...'.format(args.num_workers))
    deliver = SampleDeliver(subgraph, sub_trainnid, args.num_neighbors, hops, args.num_workers)
    deliver.async_sample(args.n_epochs, args.batch_size, one2all=args.one2all)

  print('start running graph server on dataset: {}'.format(graph_name))
  sys.stdout.flush()
  g.run()



if __name__ == '__main__':
  print("Attemting to start pa server")
  sys.stdout.flush()
  parser = argparse.ArgumentParser(description='GraphServer')

  parser.add_argument("--dataset", type=str, default="None",
                      help="dataset folder path")
  
  parser.add_argument("--num-workers", type=int, default=1,
                      help="the number of workers")
  
  parser.add_argument("--model", type=str, default="gcn",
                      help="model type for preprocessing")

  # sample options
  parser.add_argument("--sample", dest='sample', action='store_true')
  parser.set_defaults(sample=False)
  parser.add_argument("--num-neighbors", type=int, default=10)
  parser.add_argument("--gnn-layers", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=1032)
  #parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--n-epochs", type=int, default=2)
  parser.add_argument("--one2all", dest='one2all', action='store_true')
  parser.set_defaults(one2all=False)

  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  
  args = parser.parse_args()
  
  parser.set_defaults(preprocess = False)

  main(args)
