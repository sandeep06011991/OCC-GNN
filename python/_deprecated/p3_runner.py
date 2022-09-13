from random import shuffle
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
import dgl.function as fn
import tqdm
from utils.utils import thread_wrapped_func, get_process_graph
# from utils.convert_dgl_dataset import get_dataset
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
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])


def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id):
    """
    Extracts features and labels for a subset of nodes.
    """
    assert(nfeat.device == torch.device('cpu'))
    assert(labels.device == torch.device('cpu'))

    #batch_inputs = nfeat[input_nodes].to(dev_id)
    batch_inputs = None
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels

#### Entry point


def aggregate(block, feats, device_id):
    with block.local_scope():
        block.nodes['_N'].data['in'] = feats[block.srcdata['_ID']]
        block.update_all(fn.copy_u('in', 'm'),
                         fn.sum('m', 'out'))
        return block.dstdata['out']


def run(proc_id, n_gpus, args, devices, data, queues):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12346')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    th.cuda.set_device(dev_id)

    # Unpack data
    n_classes, graph, train_nfeat, train_labels = data

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(dev_id)
        train_labels = train_labels.to(dev_id)

    in_feats = train_nfeat.shape[1]

    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = ~(graph.ndata['train_mask'] | graph.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # Split train_nid
    train_nid = th.split(train_nid, math.ceil(
        len(train_nid) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    graph = graph.to(dev_id)
    train_nid = train_nid.to(dev_id)

    # Put 25% of node feature data on the gpu. send to gpu
    # slice = nfeats/num proces
    # nfeat_slice = train_nfeat[:, slice * proc_id : slice * (proc_id +1)train_nid]
    slice_len = int(train_nfeat.shape[1] / n_gpus)
    slice = train_nfeat[:, slice_len * proc_id: slice_len * (proc_id + 1)]
    proc_slice = slice.to(dev_id)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])

    dataloader = dgl.dataloading.NodeDataLoader(
        graph,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes,
                 args.num_layers-1, F.relu, args.dropout, args.deterministic)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Epoch, Forward, Backward, Shuffle
    epoch_times = []
    forward_times = []
    backward_times = []
    shuffle_times = []

    for epoch_num in range(args.num_epochs):
        if dev_id == 0:
            print('Epoch {}'.format(epoch_num))

        epoch_time = time.time()
        if epoch_num == 1:
            epoch_times = []
            forward_times = []
            backward_times = []
            shuffle_times = []

        it = enumerate(dataloader)

        while (cur := next(it, None)) is not None:
            torch.distributed.barrier()

            _, (input_nodes, seeds, blocks) = cur
            _, batch_labels = load_subtensor(train_nfeat, train_labels,
                                             seeds, input_nodes, dev_id)

            first_block = blocks[0]
            input_nodes = input_nodes.to('cpu')
            first_block = first_block.to('cpu')

            shuffle_time = time.time()

            for i in range(4):
                if i != dev_id:
                    new_block = first_block.to(i)
                    queues[i][0].put((dev_id, new_block))

            for _ in range(3):
                (to_id, first_block) = queues[dev_id][0].get()
                agg = aggregate(first_block, proc_slice, dev_id)
                queues[to_id][1].put((dev_id, agg))

            to_concat = [None]*4
            to_concat[dev_id] = aggregate(
                blocks[0].to(dev_id), proc_slice, dev_id)

            for _ in range(3):
                (from_id, res) = queues[dev_id][1].get()
                to_concat[from_id] = res.to(dev_id)

            batch_inputs = torch.concat(to_concat, dim=1)

            shuffle_times.append(time.time() - shuffle_time)

            blocks = [block.int().to(dev_id) for block in blocks[1:]]

            forward_time = time.time()
            batch_pred = model(blocks, batch_inputs)
            forward_times.append(time.time() - forward_time)

            loss = loss_fcn(batch_pred, batch_labels)

            optimizer.zero_grad()

            backward_time = time.time()
            loss.backward()
            backward_times.append(time.time() - backward_time)

            optimizer.step()

        epoch_times.append(time.time() - epoch_time)

    print('Epoch time: {}'.format(sum(epoch_times)/args.num_epochs))
    print('Forward time: {}'.format(sum(forward_times)/args.num_epochs))
    print('Backward time: {}'.format(sum(backward_times)/args.num_epochs))
    print('Shuffle time: {}'.format(sum(shuffle_times)/args.num_epochs))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph', type=str, default="ogbn-arxiv")
    argparser.add_argument('--fsize', type=int, default=-1)
    argparser.add_argument('--gpu', type=str,
                           default=0,
                           # default='0,1,2,3',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_false',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true', default=True,
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument(
        '--deterministic', default=False, action="store_true")
    args = argparser.parse_args()

    n_gpus = 4
    devices = [0, 1, 2, 3]

    n_classes = 40
    g, p_map, num_classes = get_process_graph(args.graph, args.fsize)

    train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # Pack data
    data = num_classes, g, g.ndata.pop(
        'features', None), g.ndata.pop('labels', None)
    mp.set_start_method('spawn')

    if n_gpus == 1:
        # assert(false)
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        print("Launch multiple gpus")
        queues = [[mp.Queue(),  mp.Queue()] for _ in range(n_gpus)]
        for proc_id in range(n_gpus):
            p = mp.Process(target=run,
                           args=(proc_id, n_gpus, args, devices, data, queues))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        print("DONE")
