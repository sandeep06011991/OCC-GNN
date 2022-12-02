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
# from dgl._deprecate.graph import DGLGraph
import os
import torch.multiprocessing as mp



FEAT_DICT = {"ogbn-arxiv":128, "ogbn-products":100,\
                "amazon":200, "reorder-papers100M":128, \
                    "com-orkut":400}
N_CLASSES = {"ogbn-arxiv":40, "ogbn-products":48,\
                "amazon":102, "reorder-papers100M":172, \
                    "com-orkut":48}


from utils.timing_analysis import *
from utils.env import *
from utils.utils import *
import pickle
from PaGraph.model.dgl_sage import  *
from PaGraph.model.dgl_gat import  *

import PaGraph.data as data
import PaGraph.storage as storage
from PaGraph.parallel import SampleLoader
import nvtx

def init_process(rank, world_size, backend):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(rank)
  print('rank [{}] process successfully launches'.format(rank))

username = os.environ['USER']
if username =="spolisetty_umass_edu":
    ROOT_DIR = "/work/spolisetty_umass_edu/data/pagraph"
    PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN/upgraded_pagraph"
if username == "spolisetty":
    ROOT_DIR = "/data/sandeep/pagraph"
    PATH_DIR = "/home/spolisetty/OCC-GNN/upgraded_pagraph"

path_set = False
for p in sys.path:
    print(p)
    if PATH_DIR ==  p:
       path_set = True
if (not path_set):
    print("Setting Path")
    sys.path.append(PATH_DIR)

def avg(ls):
    return (sum(ls[1:])/(len(ls)-1))

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred )
import logging


def trainer(rank, world_size, args, metrics_queue , backend='nccl'):
  dataset = "{}/{}".format(ROOT_DIR, args.dataset)
  feat_size = FEAT_DICT[args.dataset]
  # init multi process
  init_process(rank, world_size, backend)
  # load datai

  if rank == 0:
    os.makedirs('{}/logs'.format(PATH_DIR),exist_ok = True)
    FILENAME= ('{}/logs/{}_{}_{}_{}.txt'.format(PATH_DIR, \
             args.dataset, args.batch_size, int(100* (args.cache_per)),args.model))

    fileh = logging.FileHandler(FILENAME, 'w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileh.setFormatter(formatter)

    log = logging.getLogger()  # root logger
    log.addHandler(fileh)      # set the new handler
    log.setLevel(logging.INFO)


  dataname = os.path.basename(dataset)
  remote_g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")

  num_nodes, num_edges = remote_g.proxy.get_graph_info(dataname)
  num_nodes, num_edges = int(num_nodes), int(num_edges)
  adj, t2fid = data.get_sub_train_graph(dataset, rank, world_size)

  g = dgl.DGLGraph(adj, readonly=True)
  n_classes = N_CLASSES[args.dataset]
  train_nid = data.get_sub_train_nid(dataset, rank, world_size)
  print("Training_nid", train_nid.shape, rank)
  print("Expected number of minibatches",train_nid.shape[0]/args.batch_size)
  if rank == 0:
      log.info("Expected number of minibatches: {}".format(train_nid.shape[0]/args.batch_size))
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
  print("Training subgraph nodes", adj.shape[0], "Main graph nodes", num_nodes)
  cacher = storage.GraphCacheServer(remote_g, num_nodes, t2fid, rank, args.cache_per)
  cacher.init_field(embed_names)
  cacher.log = True
  heads = 3
  in_feats = feat_size
  in_dim = feat_size
  # prepare model
  num_hops = args.n_layers
  if args.model == "gcn":
      model = SAGE(in_feats, args.n_hidden, n_classes,
               args.n_layers, F.relu, args.dropout)
  else:
      assert(args.model == "gat")
      heads = 3
      model = GAT(in_feats, args.n_hidden, \
              n_classes , heads, args.n_layers, F.relu, args.dropout)

  loss_fcn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
  model.cuda(rank)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  ctx = torch.device(rank)

  sampler = dgl.dataloading.NeighborSampler(
            [int(args.num_neighbors) for i in range(3)], replace = True)
  world_size = 4
  #train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]
  number_of_minibatches = train_nid.shape[0]/args.batch_size
  g = g.add_self_loop()
  dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cpu',
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor = 2,
        num_workers= 4,
        persistent_workers= True)

  epoch_dur = []
  print("Our caching model does not require a warm up")
  tic = time.time()
  t3 = time.time()
  epoch_time = []
  miss_rate_per_epoch = []
  miss_num_per_epoch = []
  time_cache_gather = []
  event_cache_gather = []
  time_cache_move = []
  event_cache_move = []
  edges_processed = []
  e1 = torch.cuda.Event(enable_timing = True)
  e2 = torch.cuda.Event(enable_timing = True)
  if rank == 0:
    log.info("Running for epochs {}".format(args.n_epochs))
  epoch_metrics = []
  with torch.autograd.profiler.profile(enabled=(False), use_cuda=True) as prof:
    cacher.auto_cache(g,embed_names)
    for epoch in range(args.n_epochs):
      model.train()
      epoch_start_time = time.time()
      epoch_edges_processed = 0
      step = 0
      #print("start epoch",rank)
      #for nf in sampler:
      it = iter(dataloader)
      minibatch_metrics = []
      while True:
        t00 = time.time()
        batch_time = {}
        try:
            with nvtx.annotate('sample',color = 'yellow'):
                s1 = time.time()
                input_nodes, seeds, blocks = next(it)
                assert(input_nodes.shape[0] == blocks[0].number_of_src_nodes())
                assert(seeds.shape[0] == blocks[-1].number_of_dst_nodes())
                s2 = time.time()
                batch_time[SAMPLE_START_TIME] = s1
                batch_time[GRAPH_LOAD_START_TIME] = s2
        except StopIteration:
            break
        #torch.distributed.barrier()
        with nvtx.annotate("cache",color = 'blue'):
        #with torch.autograd.profiler.record_function('gpu-load'):
        #if True:
          s0 = time.time()
          blocks = [b.to(rank) for b in blocks]
          s1 = time.time()
          batch_time[DATALOAD_START_TIME] = s1
          input_data = cacher.fetch_data(input_nodes)
          batch_nids = seeds
          label = labels[batch_nids]
          label = label.cuda(rank, non_blocking=False)
          batch_time[DATALOAD_END_TIME] = time.time()
          # print("move time", e4.elapsed_time(e5)/1000)
          #print("Cache time",s2-s1)
        #with torch.autograd.profiler.record_function('gpu-compute'):
        with nvtx.annotate('compute', color = 'red'):
        #if True:
          e1.record()
          pred = model(blocks, input_data['features'])
          for i in range(3):
              epoch_edges_processed += blocks[i].num_edges()
          e2.record()
          loss = loss_fcn(pred, label)
          acc = compute_acc(pred,label)
          e2.synchronize()
          optimizer.zero_grad()
          loss.backward()
          batch_time[FORWARD_ELAPSED_EVENT_TIME] = e1.elapsed_time(e2)/1000
          optimizer.step()
          # Use this for consistency
          # torch.cuda.synchronize()
          batch_time[END_BACKWARD] = time.time()
          minibatch_metrics.append(batch_time)

        #print("Compute time without sync", e1.elapsed_time(e2)/1000)
        #forward_time_epoch += (e1.elapsed_time(e2)/1000)
        t11 = time.time()
        step += 1
        #print("current minibatch",step,rank, epoch)
        if args.end_early and step == 5:
          break
        if rank == 0:
          log.info("iteration : {}, epoch: {}, iteration time: {}".format(step, epoch, t11-t00))

            #cacher.auto_cache(g, embed_names)
        if rank == 0 and step % 20 == 0:
          print('epoch [{}] step [{}]. Loss: {:.4f}'
                .format(epoch + 1, step, loss.item()))
      epoch_dur.append(time.time() - epoch_start_time)
      if rank == 0:
        # compute_time.append(epoch_compute_time)
        # sample_time.append(epoch_sample_time)
        # log.info("epoch:{} collected_sample:{}".format(epoch, sample_time))
        # graph_move_time.append(epoch_move_graph_time)
        # log.info("epoch:{} graph move time:{}".format(epoch, graph_move_time))

        collect_c, move_c, coll_t, mov_t = cacher.get_time_and_reset_time()
        time_cache_gather.append(coll_t)
        time_cache_move.append(mov_t)
        event_cache_gather.append(collect_c)
        event_cache_move.append(move_c)
        # log.info("epoch:{}, cache gather {},cache move {}".format(epoch, time_cache_gather\
              # , time_cache_move))
        # forward_time.append(forward_time_epoch)
        # backward_time.append(backward_time_epoch)
        # log.info("epoch: {}, forward time:{}".format(epoch, forward_time))
        # log.info("epoch: {}, backward time:{}".format(epoch, backward_time))
        #print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
      miss_rate, miss_num = cacher.get_miss_rate()
      miss_rate_per_epoch.append(miss_rate)
        # Append in MB
      miss_num_per_epoch.append(miss_num * in_dim * 4 / (1024 * 1024))
        #print('Epoch average miss rate: {:.4f}'.format(miss_rate))
      edges_processed.append(epoch_edges_processed)
      epoch_metrics.append(minibatch_metrics)
    toc = time.time()
  print("Exiting training working to collect profiler results")
  print("Epoch ", epoch_dur, rank)
  if rank == 0:
      print("accuracy: {:.4f}\n".format(acc))
      # print("Sample time: {:.4f}s\n".format(avg(sample_time)))
      # print("forward time: {:.4f}s\n".format(avg(forward_time)))
      # print("backward time: {:.4f}s\n".format(avg(backward_time)))
      # print("movement graph: {:.4f}s\n".format(avg(graph_move_time)))
      # print("CPU collect: {:.4f}s\n".format(avg(time_cache_gather)))
      # print("CUDA collect: {:.4f}s\n".format(avg(event_cache_gather)))
      # print("CPU move: {:.4f}s\n".format(avg(time_cache_move)))
      # print("CUDA move: {:.4f}s\n".format(avg(event_cache_move)))
      print("Epoch time: {:.4f}s\n".format(avg(epoch_dur)))
      print("Miss rate: {:.4f}s\n".format(avg(miss_rate_per_epoch)))
  print("Miss num per epoch: {:.4f}MB, device {}\n".format(int(avg(miss_num_per_epoch)),rank))
  print("Edges processed per epoch: {}".format(avg(edges_processed)))
  with open('metrics{}.pkl'.format(rank), 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(epoch_metrics, outp, pickle.HIGHEST_PROTOCOL)

  # Profiling everything is unstable and overweight.
  # torch profiler uses events under it.
  #if rank == 0:
  #  print(prof.key_averages().table(sort_by='cuda_time_total'))
  print('Total Time: {:.4f}s'.format(toc - tic))
  if rank == 0:
    remote_g.destroy()
  else:
    remote_g.proxy = None

  print("All cleaned up")

def train_hook(*args):
  # import line_profiler
  from line_profiler import LineProfiler
  prof = LineProfiler()
  train_w = prof(trainer)
  train_w(*args)
  prof.dump_stats('train.lprof')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')

  parser.add_argument("--gpu", type=str, default='0,1,2,3',
                      help="gpu ids. such as 0 or 0,1,2")
  parser.add_argument("--dataset", type=str, default="None", required = True,
                      help="path to the dataset folder")
  # model arch
  #parser.add_argument("--feat-size", type=int,
  #                    help='input feature size')
  parser.add_argument("--n-classes", type=int, default=60)
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="dropout probability")
  parser.add_argument("--n-hidden", type=int, default=16,
                      help="number of hidden gcn units")
  parser.add_argument("--n-layers", type=int, default=3,
                      help="number of hidden gcn layers")
  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  # training hyper-params
  parser.add_argument("--lr", type=float, default=3e-2,
                      help="learning rate")
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
  parser.add_argument("--model",type = str, required = True)
  parser.add_argument("--cache-per", type = float, required = True)
  parser.add_argument("--end-early", help="increase output verbosity",
                    action="store_true")
  parser.set_defaults(remote_sample=False)

  args = parser.parse_args()
  metrics_queue = mp.Queue(4)
  # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  # gpu_num = len(args.gpu.split(','))
  gpu_num = 4

  mp.spawn(trainer, args=(gpu_num, args, metrics_queue), nprocs=gpu_num, join=True)
  collected_metrics = []
  for i in range(4):
    with open("metrics{}.pkl".format(i), "rb") as input_file:
      cm = pickle.load(input_file)
    collected_metrics.append(cm)

  epoch_batch_sample, epoch_batch_graph, epoch_batch_feat_time, \
      epoch_batch_forward, epoch_batch_backward, \
      epoch_batch_loadtime, epoch_batch_totaltime = \
          compute_metrics(collected_metrics)
  print("sample_time:{}".format(epoch_batch_sample))
  print("data movement:{}".format(epoch_batch_loadtime))
  print("movement graph:{}".format(epoch_batch_graph))
  print("movement feature:{}".format(epoch_batch_loadtime))
  print("forward time:{}".format(epoch_batch_forward))
  print("backward time:{}".format(epoch_batch_backward))
