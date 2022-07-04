import dgl
import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm
from utils.utils import get_process_graph
from utils.utils import thread_wrapped_func
# from load_graph import load_reddit, inductive_split

from models.sage import SAGE

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

class PaCache:

    def __init__(self,graph,nfeat,max_size,dev_id):
            self.graph = graph
            self.max_size = max_size
            self.dev_id = dev_id
            # Notes on  why in-degree measures greatest likely hood.
            # indptr degrees are out_degrees.
            # Hence node with highest in_degree will be sampled in first layer
            # g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
            # g.in_degrees()
            # tensor([0, 1, 1, 1, 1, 1])
            # g.out_degrees()
            # tensor([5, 0, 0, 0, 0, 0])
            # g.adj(scipy_fmt = 'csr').indptr
            # array([0, 5, 5, 5, 5, 5, 5], dtype=int32)
            in_degree = self.graph.in_degree(v='__ALL__')
            values, indices = torch.sort(in_degree, descending = True)
            assert(max_size <= 1 and max_size >=0)
            print("cache size in GB:",max_size * graph.num_nodes() * nfeat.shape[1] * 4 / (1024 * 1024* 1024))
            self.cached_indices = indices[:int(max_size * graph.num_nodes())].to('cpu')
            self.cache_mask = torch.zeros(graph.num_nodes(),dtype= bool)
            self.cache_mask[self.cached_indices] = True
            self.cache_index = torch.zeros(graph.num_nodes(),dtype = torch.long)
            self.cache_index[self.cached_indices] = torch.range(0,max(0,self.cached_indices.shape[0]-1),dtype = torch.long)
            self.tocache = nfeat[self.cached_indices].to(dev_id)
            self.avg_cache_hit_rate = []
    # @profile
    def load_subtensor(self,nfeat, labels, seeds, input_nodes, dev_id):
        """
        Extracts features and labels for a subset of nodes.
        """
        # We measure total time including cost of processing cache values.
        t1 = time.time()
        input_index = torch.where(self.cache_mask[input_nodes])[0]
        cache_hit = input_nodes[input_index]
        cache_index = self.cache_index[cache_hit]
        assert(input_index.shape == cache_index.shape)
        # cache_hit,input_index,cache_index = np.intersect1d(input_nodes, self.cached_indices,return_indices = True)
        self.avg_cache_hit_rate.append(cache_hit.shape[0]/ input_nodes.shape[0])
        if(cache_hit.shape[0] == 0):
            print("no cache hit")
            batch_inputs = nfeat[input_nodes].to(dev_id)
        else:
            # print("cache hit",input_index.shape[0]/input_nodes.shape[0])
            cache_mask = torch.zeros(input_nodes.shape[0],dtype = torch.bool)
            cache_mask [input_index] = True
            batch_inputs = torch.cuda.FloatTensor(input_nodes.shape[0],nfeat.shape[1],device=dev_id)
            batch_inputs[input_index] = self.tocache[cache_index]
            cpu_indices = torch.where(~ cache_mask)[0]
            # print(cpu_indices)
            batch_inputs[cpu_indices.to(dev_id) ] = nfeat[input_nodes[cpu_indices]].to(dev_id)
        t2 = time.time()
        # batch_inputs = nfeat[input_nodes].to(dev_id)
        batch_labels = labels[seeds].to(dev_id)

        #print("cache miss time {} and bandwidth {} for device: {}".format(t2-t1, \
        #        (input_nodes.shape[0] - cache_hit.shape[0])/((t2-t1) *(1024 * 1024 * 1024)), dev_id))
        return batch_inputs, batch_labels, t2-t1

#### Entry point

def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    out = th.cuda.set_device(dev_id)
    # Unpack data
    n_classes, train_g, val_g, test_g = data

    if args.inductive:
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')
    train_nfeat = train_nfeat.pin_memory()
    # if not args.data_cpu:
    #     train_nfeat = train_nfeat.to(dev_id)
    #     train_labels = train_labels.to(dev_id)
    assert(train_nfeat.device == torch.device('cpu'))
    in_feats = train_nfeat.shape[1]

    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    #train_nid =torch.arange(train_g.num_nodes())
    #print("Num Nodes",len(train_nid), train_g.num_nodes())
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    # Split train_nid
    print(train_nid.shape,"Train nid", len(train_nid), n_gpus)
    train_nid = th.split(train_nid, math.ceil(len(train_nid) / n_gpus))[proc_id]
    print(train_nid.shape, "trainin nid split")
    # Create PyTorch DataLoader for constructing blocks
    # train_g = train_g.to(dev_id)
    # train_nid = train_nid.to(dev_id)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    ##### print("batch_size:",args.batch_size)
    ##### Cache #######
    assert(args.cache_per <= 1)
    print("Begin cache")
    cache = PaCache(train_g,train_nfeat, args.cache_per ,dev_id)
    print("End cache")
    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_nodes = train_g.num_nodes()

    # Training loop
    epoch_time = []
    iter_tput = []
    move_time = []
    forward_backward_time = []
    forward_time_epoch = []
    back_time_epoch = []
    sample_time = []
    ii = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for epoch in range(args.num_epochs):
        tic = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        it = enumerate(dataloader)
        move_time_epoch = 0
        forward_backward_time_epoch = 0
        forward_time = 0
        back_time = 0
        nodes_done = 0
        sample_get_time = 0
        while(True):
            try:
                t0 = time.time()
                step, (input_nodes, seeds, blocks) = next(it)
                nodes_done += seeds.shape[0]
                t1 = time.time()
                sample_get_time += (t1 - t0)
                batch_inputs, batch_labels,diff_time = cache.load_subtensor(train_nfeat, train_labels, seeds, input_nodes, dev_id)
                blocks = [block.int().to(dev_id) for block in blocks]
                t2 = time.time()

                #print("total memory management time time ", t2 -t1)
                #print("cache tranfer time", diff_time)

                move_time_epoch += (diff_time)
                t3 = time.time()
                start.record()
                batch_pred = model(blocks, batch_inputs)
                t33 = time.time()
                loss = loss_fcn(batch_pred, batch_labels)
                # if dev_id == 0:
                #     print("accuracy",\
                #         torch.sum(torch.max(batch_pred,1)[1]==batch_labels)/batch_pred.shape[0])
                #print("loss",loss)
                optimizer.zero_grad()
                loss.backward()
                end.record()
                t4 = time.time()
                torch.cuda.synchronize(end)
                #print("forward backward time",t4-t3)
                # if proc_id == 0:
                #     print("forward back time with cuda timers",start.elapsed_time(end)/1000)
                forward_backward_time_epoch += start.elapsed_time(end)/1000
                forward_time += (t33 - t3)
                back_time += (t4-t33)
                # forward_backward_time_epoch += (t4 - t3)
                optimizer.step()
                    # break
                #torch.cuda.nvtx.range_pop()
            except StopIteration:
                break
                #torch.cuda.nvtx.range_pop()
            # if step % args.log_every == 0 and proc_id == 0:
            #     acc = compute_acc(batch_pred, batch_labels)
            #     print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
            #         epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))
        print("Exiting main training loop")
        if n_gpus > 1:
            th.distributed.barrier()
        toc = time.time()
        sample_time.append(sample_get_time)
        epoch_time.append(toc-tic)
        move_time.append(move_time_epoch)
        forward_backward_time.append(forward_backward_time_epoch)
        forward_time_epoch.append(forward_time)
        back_time_epoch.append(back_time)
        # if proc_id == 0:
        #     print('Epoch Time(s): {:.4f}'.format(toc - tic))
        #     if epoch >= 5:
        #         avg += toc - tic
        #     if epoch % args.eval_every == 0 and epoch != 0:
        #         if n_gpus == 1:
        #             eval_acc = evaluate(
        #                 model, val_g, val_nfeat, val_labels, val_nid, devices[0])
        #             test_acc = evaluate(
        #                 model, test_g, test_nfeat, test_labels, test_nid, devices[0])
        #         else:
        #             eval_acc = evaluate(
        #                 model.module, val_g, val_nfeat, val_labels, val_nid, devices[0])
        #             test_acc = evaluate(
        #                 model.module, test_g, test_nfeat, test_labels, test_nid, devices[0])
        #         print('Eval Acc {:.4f}'.format(eval_acc))
        #         print('Test Acc: {:.4f}'.format(test_acc))

    num_epochs = args.num_epochs
    if n_gpus > 1:
        th.distributed.barrier()
    print("forward_time_epoch",forward_time_epoch)
    print("back_time_epoch",back_time_epoch)
    print("forward_backward_time_epoch",forward_backward_time_epoch)
    if proc_id == 0:
        # assert(len(forward_time) == num_epochs)
        #if True:
        assert(num_epochs > 1)
        print("avg cache hit rate: {}".format(sum(cache.avg_cache_hit_rate)\
               /len(cache.avg_cache_hit_rate)))

        # print("Avg forward backward time: {}sec, device {}".format(sum(forward_time[1:])/(num_epochs - 1), dev_id))


        print("avg move time: {}sec, device {}".format(sum(move_time[1:])/(num_epochs - 1), dev_id))
        print(move_time)
        print('avg epoch time: {}sec, device {}'.format(sum(epoch_time[1:])/(num_epochs - 1), dev_id))
        print(epoch_time)
        print('avg sample time:{}sec'.format(sample_time))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, default = "ogbn-arxiv")
    argparser.add_argument('--fsize', type = int, default = -1 , help = "fsize only for synthetic graphs")
    argparser.add_argument('--cache-per', type = float, default = .25)
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,25')
    argparser.add_argument('--batch-size', type=int, default=1032)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--num-workers', type=int, default=1,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_false',
                           help="Inductive learning setting")
    args = argparser.parse_args()
    n_gpus = 4
    devices = [0,1,2,3]
    #n_gpus = 1
    #devices = [2]
    # devices = list(map(int, args.gpu.split(',')))
    # n_gpus = len(devices)
    # assert(n_gpus > 0)
    # print(n_gpus,devices)
    g,p_map,num_classes =  get_process_graph(args.graph, args.fsize)
    args.inductive = False
    # if args.inductive:
    #     train_g, val_g, test_g = inductive_split(g)
    # else:
    train_g = val_g = test_g = g
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    data = num_classes, train_g, val_g, test_g
    start_time = time.time()
    # assert(False)
    if n_gpus == 1:
        # assert(false)
        print("Running on single GPUs")
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        print("Launch multiple gpus")
        for proc_id in range(n_gpus):
            p = mp.Process(target=thread_wrapped_func(run),
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    end_time = time.time()
    print("Total time across all processes", end_time - start_time)
