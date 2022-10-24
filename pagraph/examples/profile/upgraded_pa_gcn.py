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

PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN/pagraph"
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


from PaGraph.model.gcn_nssc import GCNSampling
from PaGraph.model.gat_nodeflow import GATNodeFlow
import PaGraph.data as data
import PaGraph.storage as upgraded_storage
from PaGraph.parallel import SampleLoader
import nvtx
def init_process(rank, world_size, backend):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29501'
  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(rank)
  print('rank [{}] process successfully launches'.format(rank))

ROOT_DIR = "/work/spolisetty_umass_edu/data/pagraph"

def avg(ls):
    return (sum(ls[1:])/(len(ls)-1))

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred )
import logging


def trainer(rank, world_size, args, backend='nccl'):
  dataset = "{}/{}/".format(ROOT_DIR, args.dataset)
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

  num_nodes, num_edges, _, _ = remote_g.proxy.get_graph_info(dataname)
  num_nodes, num_edges = int(num_nodes), int(num_edges)
  adj, t2fid = data.get_sub_train_graph(dataset, rank, world_size)

  g = DGLGraph(adj, readonly=True)
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

  # to torch tensor
  t2fid = torch.LongTensor(t2fid)
  labels = torch.LongTensor(labels)
  embed_names = ['features', 'norm']
  print("Training subgraph nodes", adj.shape[0], "Main graph nodes", num_nodes)
  cacher = storage.GraphCacheServer(remote_g, adj.shape[0], t2fid, rank, args.cache_per)
  cacher.init_field(embed_names)
  cacher.log = True
  
  # prepare model
  num_hops = args.n_layers if args.preprocess else args.n_layers + 1
  if args.model == "gcn":
      in_dim = feat_size
      model = GCNSampling(feat_size,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.preprocess)
  else:
      assert(args.model == "gat")
      residual = False
      num_layers = args.n_layers
      in_dim = feat_size
      num_hidden = args.n_hidden
      num_classes = n_classes
      num_heads = 3
      feat_drop = args.dropout
      attn_drop = args.dropout
      model = GATNodeFlow(
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 residual,
                 activation=F.relu)

  loss_fcn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
  model.cuda(rank)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  ctx = torch.device(rank)

  if args.remote_sample:
    sampler = SampleLoader(g, rank, one2all=False)
  else:
    sampler = dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
      args.num_neighbors, neighbor_type='in',
      shuffle=True, num_workers=args.num_workers,
      num_hops=num_hops, seed_nodes=train_nid,
      prefetch=True
    )
  # start training
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
  forward_time = []
  backward_time = []
  sample_time = []
  graph_move_time = []
  edges_processed = []
  e1 = torch.cuda.Event(enable_timing = True)
  e2 = torch.cuda.Event(enable_timing = True)
  e3 = torch.cuda.Event(enable_timing = True)
  e4 = torch.cuda.Event(enable_timing = True)
  e5 = torch.cuda.Event(enable_timing = True)
  if rank == 0:
    log.info("Running for epochs {}".format(args.n_epochs))
  with torch.autograd.profiler.profile(enabled=(False), use_cuda=True) as prof:
    cacher.auto_cache(g,embed_names)
    for epoch in range(args.n_epochs):
      model.train()
      epoch_start_time = time.time()
      forward_time_epoch = 0
      backward_time_epoch = 0
      epoch_sample_time = 0
      epoch_move_graph_time = 0
      epoch_edges_processed = 0
      step = 0
      #print("start epoch",rank)
      #for nf in sampler:
      it = iter(sampler)
      while True:
        t00 = time.time()
        try:
            with nvtx.annotate('sample',color = 'yellow'):
                s1 = time.time()
                nf = next(it)
                s2 = time.time()
                epoch_sample_time += (s2 - s1)
        except StopIteration:
            break
        print("sample time", s2-s1) 
        #torch.distributed.barrier()
        with nvtx.annotate("cache",color = 'blue'):
        #with torch.autograd.profiler.record_function('gpu-load'):
        #if True:
          s0 = time.time()
          e4.record()
          nf.copy_from_parent(ctx)
          s1 = time.time()
          e5.record()
          cacher.fetch_data(nf)
          batch_nids = nf.layer_parent_nid(-1)
          label = labels[batch_nids]
          label = label.cuda(rank, non_blocking=True)
          e4.synchronize()
          epoch_move_graph_time += max((s1 - s0),e4.elapsed_time(e5)/1000)
          print("move time", e4.elapsed_time(e5)/1000)
          #print("Cache time",s2-s1)
        e1.record()
        #with torch.autograd.profiler.record_function('gpu-compute'):
        with nvtx.annotate('compute', color = 'red'):
        #if True:
          pred = model(nf)
          for i in range(3):
              epoch_edges_processed += nf.block_size(i)
          print("edges:" , epoch_edges_processed)

          e2.record()
          loss = loss_fcn(pred, label)
          acc = compute_acc(pred,label)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          e3.record()
        e3.synchronize()
        print("Training time", e1.elapsed_time(e3)/1000)
        #print("Compute time without sync", e1.elapsed_time(e2)/1000)
        forward_time_epoch += (e1.elapsed_time(e2)/1000)
        backward_time_epoch += (e2.elapsed_time(e3)/1000)
        t11 = time.time()
        step += 1
        #print("current minibatch",step,rank, epoch)
        if args.end_early and step == 5:
          break
        if rank == 0:
          log.info("iteration : {}, epoch: {}, iteration time: {}".format(step, epoch, t11-t00)) 
    
        if epoch == 0 and step == 1:
          pass
            #cacher.auto_cache(g, embed_names)
        if rank == 0 and step % 20 == 0:
          print('epoch [{}] step [{}]. Loss: {:.4f}'
                .format(epoch + 1, step, loss.item()))
      if rank == 0:
        # compute_time.append(epoch_compute_time)
        sample_time.append(epoch_sample_time)
        log.info("epoch:{} collected_sample:{}".format(epoch, sample_time))
        graph_move_time.append(epoch_move_graph_time)
        log.info("epoch:{} graph move time:{}".format(epoch, graph_move_time))
        epoch_dur.append(time.time() - epoch_start_time)
        collect_c, move_c, coll_t, mov_t = cacher.get_time_and_reset_time()
        time_cache_gather.append(coll_t)
        time_cache_move.append(mov_t)
        event_cache_gather.append(collect_c)
        event_cache_move.append(move_c)
        log.info("epoch:{}, cache gather {},cache move {}".format(epoch, time_cache_gather\
                , time_cache_move))
        forward_time.append(forward_time_epoch)
        backward_time.append(backward_time_epoch)
        log.info("epoch: {}, forward time:{}".format(epoch, forward_time))
        log.info("epoch: {}, backward time:{}".format(epoch, backward_time))
        #print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
        miss_rate, miss_num = cacher.get_miss_rate()
        miss_rate_per_epoch.append(miss_rate)
        # Append in MB
        miss_num_per_epoch.append(miss_num * in_dim * 4 / (1024 * 1024))
        #print('Epoch average miss rate: {:.4f}'.format(miss_rate))
        edges_processed.append(epoch_edges_processed)
        log.info("miss num: {}".format(miss_num_per_epoch))
        log.info("edges processed: {}".format(edges_processed))
        
    toc = time.time()
  print("Exiting training working to collect profiler results")
  if rank == 0:
      print("accuracy: {:.4f}\n".format(acc))
      print("Sample time: {:.4f}s\n".format(avg(sample_time)))
      print("forward time: {:.4f}s\n".format(avg(forward_time)))
      print("backward time: {:.4f}s\n".format(avg(backward_time)))
      print("movement graph: {:.4f}s\n".format(avg(graph_move_time)))
      print("CPU collect: {:.4f}s\n".format(avg(time_cache_gather)))
      print("CUDA collect: {:.4f}s\n".format(avg(event_cache_gather)))
      print("CPU move: {:.4f}s\n".format(avg(time_cache_move)))
      print("CUDA move: {:.4f}s\n".format(avg(event_cache_move)))
      print("Epoch time: {:.4f}s\n".format(avg(epoch_dur)))
      print("Miss rate: {:.4f}s\n".format(avg(miss_rate_per_epoch)))
      print("Miss num per epoch: {:.4f}MB, device {}\n".format(int(avg(miss_num_per_epoch)),rank))
      print("Edges processed per epoch: {}".format(avg(edges_processed)))
  # Profiling everything is unstable and overweight.
  # torch profiler uses events under it.
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
  parser.add_argument("--n-layers", type=int, default=2,
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

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpu_num = len(args.gpu.split(','))

  mp.spawn(trainer, args=(gpu_num, args), nprocs=gpu_num, join=True)
import os
import sys
import argparse, time
import torch
