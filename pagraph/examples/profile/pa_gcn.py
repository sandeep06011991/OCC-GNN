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
from dgl import DGLGraph

PATH_DIR = "/home/spolisetty_umass_edu/OCC-GNN/pagraph"
path_set = False
for p in sys.path:
    print(p)
    if PATH_DIR ==  p:
       path_set = True
if (not path_set):
    print("Setting Path")
    sys.path.append(PATH_DIR)


from PaGraph.model.gcn_nssc import GCNSampling
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

ROOT_DIR = "/work/spolisetty_umass_edu/pagraph"

def avg(ls):
    return (sum(ls[1:])/(len(ls)-1))

def trainer(rank, world_size, args, backend='nccl'):
  dataset = "{}/{}/".format(ROOT_DIR, args.dataset)

  # init multi process
  init_process(rank, world_size, backend)
  # load data
  dataname = os.path.basename(dataset)
  remote_g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")

  adj, t2fid = data.get_sub_train_graph(dataset, rank, world_size)
  g = DGLGraph(adj, readonly=True)
  n_classes = args.n_classes
  train_nid = data.get_sub_train_nid(dataset, rank, world_size)
  print("Training_nid", train_nid.shape, rank)
  print("Expected number of minibatches",train_nid.shape[0]/args.batch_size)
  sub_labels = data.get_sub_train_labels(dataset, rank, world_size)
  labels = np.zeros(np.max(train_nid) + 1, dtype=np.int)
  labels[train_nid] = sub_labels.flatten()

  # to torch tensor
  t2fid = torch.LongTensor(t2fid)
  labels = torch.LongTensor(labels)
  embed_names = ['features', 'norm']
  cacher = storage.GraphCacheServer(remote_g, adj.shape[0], t2fid, rank)
  cacher.init_field(embed_names)
  cacher.log = True

  # prepare model
  num_hops = args.n_layers if args.preprocess else args.n_layers + 1
  model = GCNSampling(args.feat_size,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.preprocess)
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
  time_cache_gather = []
  event_cache_gather = []
  time_cache_move = []
  event_cache_move = []
  foward_time = []
  backward_time = []
  sample_time = []
  e1 = torch.cuda.Event(enable_timing = True)
  e2 = torch.cuda.Event(enable_timing = True)
  e3 = torch.cuda.Event(enable_timing = True)
  with torch.autograd.profiler.profile(enabled=(False), use_cuda=True) as prof:
    cacher.auto_cache(g,embed_names)
    for epoch in range(args.n_epochs):
      model.train()
      epoch_start_time = time.time()
      forward_time_epoch = 0
      backward_time_epoch = 0
      epoch_sample_time = 0
      epoch_move_graph_time = 0

      step = 0
      #print("start epoch",rank)
      #for nf in sampler:
      it = iter(sampler)
      while True:
        try:
            with nvtx.annotate('sample',color = 'yellow'):
                s1 = time.time()
                nf = next(it)
                s2 = time.time()
                epoch_sample_time += (s2 - s1)
        except StopIteration:
            break
        #torch.distributed.barrier()
        with nvtx.annotate("cache",color = 'blue'):
        #with torch.autograd.profiler.record_function('gpu-load'):
        #if True:
          s1 = time.time()
          cacher.fetch_data(nf)
          batch_nids = nf.layer_parent_nid(-1)
          label = labels[batch_nids]
          label = label.cuda(rank, non_blocking=True)
          s2 = time.time()
          nf.copy_from_parent(ctx)
          s3 = time.time()
          epoch_move_graph_time = s3 - s2
          #print("Cache time",s2-s1)
        e1.record()
        #with torch.autograd.profiler.record_function('gpu-compute'):
        with nvtx.annotate('compute', color = 'red'):
        #if True:
          pred = model(nf)
          e2.record()
          loss = loss_fcn(pred, label)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          e3.record()
        e3.synchronize()
        #print("Compute time without sync", e1.elapsed_time(e2)/1000)
        forward_time_epoch += (e1.elapsed_time(e2)/1000)
        backward_time_epoch += (e2.elapsed_time(e3)/1000)

        step += 1
        #print("current minibatch",step,rank)
        if epoch == 0 and step == 1:
            pass
            #cacher.auto_cache(g, embed_names)
        if rank == 0 and step % 20 == 0:
          print('epoch [{}] step [{}]. Loss: {:.4f}'
                .format(epoch + 1, step, loss.item()))
      if rank == 0:
        # compute_time.append(epoch_compute_time)
        sample_time.append(epoch_sample_time)
        graph_move_time.append(epoch_move_graph_time)
        epoch_dur.append(time.time() - epoch_start_time)
        collect_c, move_c, coll_t, mov_t = cacher.get_time_and_reset_time()
        time_cache_gather.append(coll_t)
        time_cache_move.append(mov_t)
        event_cache_gather.append(collect_c)
        event_cache_move.append(move_c)
        forward_time.append(forward_time_epoch)
        backward_time.append(backward_time_epoch)
        #print('Epoch average time: {:.4f}'.format(np.mean(np.array(epoch_dur[2:]))))
        miss_rate = cacher.get_miss_rate()
        miss_rate_per_epoch.append(miss_rate)
        #print('Epoch average miss rate: {:.4f}'.format(miss_rate))
    toc = time.time()
  print("Exiting training working to collect profiler results")
  if rank == 0:
      print("Sample time: {:.4}s\n".format(avg(sample_time)))
      print("forward time: {:.4}s\n".format(avg(forward_time)))
      print("backward time: {:.4}s\n".format(avg(backward_time)))
      print("CPU collect: {:.4}s\n".format(avg(time_cache_gather)))
      print("CUDA collect: {:.4}s\n".format(avg(event_cache_gather)))
      print("CPU move: {:.4}s\n".format(avg(time_cache_move)))
      print("CUDA move: {:.4}s\n".format(avg(event_cache_move)))
      print("Epoch time: {:.4}s\n".format(avg(epoch_dur)))
      print("Miss rate: {:.4}s\n".format(avg(miss_rate_per_epoch)))

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
  parser.add_argument("--dataset", type=str, default="None",
                      help="path to the dataset folder")
  # model arch
  parser.add_argument("--feat-size", type=int, default=100,
                      help='input feature size')
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
  parser.add_argument("--num-neighbors", type=int, default=10,
                      help="number of neighbors to be sampled")
  parser.add_argument("--num-workers", type=int, default=1)
  parser.add_argument("--remote-sample", dest='remote_sample', action='store_true')
  parser.set_defaults(remote_sample=False)

  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpu_num = len(args.gpu.split(','))

  mp.spawn(trainer, args=(gpu_num, args), nprocs=gpu_num, join=True)
