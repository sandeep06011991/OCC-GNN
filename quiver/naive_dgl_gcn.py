# Reaches around 0.7866 ± 0.0041 test accuracy.
import logging
import dgl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
import multiprocessing
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import torch.distributed as dist
import torch.autograd.profiler as profiler

from dgl_sage import SAGE
from dgl_gat import GAT
from utils.timing_analysis import *
#from utils.env import *
from utils.utils import *
import pickle
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
def average(ls):
    if(len(ls) == 1):
        return ls[0]
    if (len(ls) < 3):
        return sum(ls)/2
    a = max(ls[1:])
    b = min(ls[1:])
    return (sum(ls[1:]) -a -b)/(len(ls)-3)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.module.inference(g, nfeat, device)
    return compute_acc(pred[test_nid], labels[test_nid].to(device))
    # model.train()
    # return compute_acc(pred[test_nid], labels[test_nid].to(device))


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a set of nodes.
    """
    assert(nfeat.device == torch.device('cpu'))
    assert(labels.device == torch.device('cpu'))
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


# Entry point
def run(rank, args,  data):
    # Unpack data
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11112'
    dist.init_process_group('nccl', rank=rank, world_size=4)

    device = rank
    torch.cuda.set_device(device)
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g,test_graph,metrics_queue = data
    print(rank, PATH_DIR)
    if rank == 0:
      print(PATH_DIR)
      os.makedirs('{}/quiver/logs_naive'.format(ROOT_DIR),exist_ok = True)
      FILENAME= ('{}/quiver/logs_naive/{}_{}_{}.txt'.format(ROOT_DIR, \
               args.graph, args.batch_size, args.model))

      fileh = logging.FileHandler(FILENAME, 'w')
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      fileh.setFormatter(formatter)
      log = logging.getLogger()  # root logger
      log.addHandler(fileh)      # set the new handler
      log.setLevel(logging.INFO)
    print(metrics_queue)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')], replace = True)
    #sampler = dgl.dataloading.MultiLayerNeighborSampler([-1,-1,-1])

    world_size = 4
    train_nid = train_nid.split(train_nid.size(0) // world_size)[rank]

    number_of_minibatches = train_nid.shape[0]/args.batch_size

    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cpu',
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor = 2,
        num_workers = 0 if args.sample_gpu else args.num_workers,
        persistent_workers=not args.sample_gpu)

    # Define model and optimizer
    if args.model == "GCN":
        model = SAGE(in_feats, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    else:
        assert(args.model == "GAT")
        heads = 3
        model = GAT(in_feats, args.num_hidden, \
                n_classes , heads, args.num_layers, F.relu, args.dropout)

    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    best_val_acc = final_test_acc = 0
    print("start training")

    epoch_time = []
    epoch_metrics = []
    accuracy = []
    edges_per_epoch = []
    data_movement_epoch = []
    e3 = torch.cuda.Event(enable_timing = True)
    e4 = torch.cuda.Event(enable_timing = True)
    e5 = torch.cuda.Event(enable_timing = True)
    test_accuracy_list = []
    #with profile(enabled = False, activities = [ProfilerActivity.CUDA, ProfilerActivity.CPU],\
    #            use_cuda = True, \
    #            schedule = torch.profiler.schedule(wait = 2, warmup = 1, active = 1000)) as prof:
    if True:    
        for epoch in range(args.num_epochs):
            tic = time.time()

            model.train()
            #pbar = tqdm(total=train_nid.size(0))
            #pbar.set_description(f'Epoch {epoch:02d}')
            total_loss = total_correct = 0
            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            epoch_start = time.time()
            dataloader_i = iter(dataloader)
            edges_computed = 0
            data_movement = 0
            step = 0
            # For metrics collection
            minibatch_metrics = []
            try:
                while True:
                    batch_time = {}
                    optimizer.zero_grad()
                    t1 = time.time()
                    with profiler.record_function("sampling"):    
                        input_nodes, seeds, blocks = next(dataloader_i)
                    t2 = time.time()
                    batch_time[SAMPLE_START_TIME] = t1
                    print("Sample time {}|".format(device), t1)
                    batch_time[GRAPH_LOAD_START_TIME] = t2
                    print("Graph load time {}|".format(device), t2)
                # copy block to gpu
                    with profiler.record_function("movement"):
                        blocks = [blk.to(device) for blk in blocks]
                #t2 = time.time()
                # blocks = [blk.formats(['coo','csr','csc']) for blk in blocks]
                    for blk in blocks:
                        # blk.create_formats_()
                        edges_computed += blk.edges()[0].shape[0]
                #print(edges_computed)
                    t3 = time.time()
                    batch_time[DATALOAD_START_TIME] = t3
                    print("movement start {}|".format(device), t3)
                # Load the input features as well as output labels
                    with profiler.record_function("movement"):
                        batch_inputs, batch_labels = load_subtensor(
                        nfeat, labels, seeds, input_nodes, device)
                    t4 = time.time()
                    batch_time[DATALOAD_END_TIME] = t4
                    print("movement end {}|".format(device), t4)
                # Compute loss and prediction
                #with torch.autograd.profiler.profile(enabled=(False), use_cuda=True, profile_memory = True) as prof:
                #if True:
                    e3.record()
                    with profiler.record_function("training"):
                        batch_pred = model(blocks, batch_inputs)
                        loss = loss_fcn(batch_pred, batch_labels)
                        e4.record()
                    #e3.record()
                        loss.backward()
                        e5.record()
                        e5.synchronize()
                        batch_time[END_BACKWARD] = time.time()
                        

                        print("Backwarwd end {}| ".format(device), batch_time[END_BACKWARD])
                        optimizer.step()
                #print("sample time", t2-t1, t3-t2)
                    batch_time[FORWARD_ELAPSED_EVENT_TIME] = e3.elapsed_time(e4)/1000
                    print("Forward time {}|".format(device), e3.elapsed_time(e4)/1000)
                #print("Time feature", device, e1.elapsed_time(e2)/1000)
                #print("Expected bandwidth", missed * nfeat.shape[1] * 4/ ((e1.elapsed_time(e2)/1000) * 1024 * 1024 * 1024), "GB device", rank, "cache rate", hit/(hit + missed))
                    data_movement += (batch_inputs.shape[0] * nfeat.shape[1] * 4/(1024 * 1024))
                    minibatch_metrics.append(batch_time)
                    if args.early_stopping and step ==5:
                        break
                    step = step + 1
                    if rank == 0:
                        log.info("step {}, epoch {}, number_of_minibatches {}".format(step, epoch, number_of_minibatches))

                    total_loss += loss.item()
                    total_correct += batch_pred.argmax(dim=-1).eq(batch_labels).sum().item()
        #        pbar.update(args.batch_size)
            except StopIteration:
                pass
            epoch_metrics.append(minibatch_metrics)
            if test_graph != None and rank== 0:
                test_nid = torch.where(test_graph.ndata['test_mask'])[0]
                test_accuracy = evaluate(model, test_graph, test_graph.ndata['features'], test_graph.ndata['labels'], \
                        test_nid, device)
                test_accuracy_list.append(test_accuracy.item())
                print("Test Accuracy:{} Epoch:{}".format(test_accuracy, epoch))
                #print("Accuracy: {}, device:{}, epoch:{}".format(test_accuracy, device, epoch))
            #prof.step()

            epoch_time.append(time.time() - epoch_start)
        #pbar.close()
            data_movement_epoch.append(data_movement)
            edges_per_epoch.append(edges_computed)
            loss = total_loss / len(dataloader)
            approx_acc = total_correct / (len(dataloader) * args.batch_size)
            accuracy.append(approx_acc)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}, Epoch Time: {time.time() - tic:.4f}')
    #print("PROFF #############")
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    if rank == 0 :
        print("Test Accuracy ###########", test_accuracy_list)
        #
        # if epoch >= 10:
        #     val_acc, test_acc = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
        #     print(f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        #
        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         final_test_acc = test_acc
    # print("edges per epoch:{}".format(average(edges_per_epoch)))
    # print("movement", device, data_movement_epoch)

    with open('metrics{}.pkl'.format(rank), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(epoch_metrics, outp, pickle.HIGHEST_PROTOCOL)

    if rank == 0:
        print("Epoch", epoch_time)
        print("accuracy:{}".format(accuracy[-1]))
        print("epoch_time:{}".format(average(epoch_time)))

        print("edges_per_epoch:{}".format(average(edges_per_epoch)))
        print("data moved:{}MB".format(average(data_movement_epoch)))
    #torch.distributed.barrier()
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_gpu_time_total', row_limit=5))

    #dist.destroy_process_group()
    return final_test_acc

if __name__ == '__main__':
    mp.set_start_method('spawn')
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--graph',type = str, required = True)
    argparser.add_argument('--model',type = str, required = True)
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=6)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='20,20,20')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default = 0)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--sample-gpu', action='store_true')
    argparser.add_argument('--early-stopping', action = 'store_true')
    argparser.add_argument('--test-graph',type = str)
    args = argparser.parse_args()
    assert args.model in ["GCN","GAT"]
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')


    # load ogbn-products data
    #root = "/mnt/bigdata/sandeep/"
    dg_graph, partition_map, num_classes = get_process_graph(args.graph, -1)
    data = dg_graph
    train_idx = torch.where(data.ndata.pop('train_mask'))[0]
    val_idx = torch.where(data.ndata.pop('val_mask'))[0]
    test_idx = torch.where(data.ndata.pop('test_mask'))[0]
    graph = dg_graph
    labels = dg_graph.ndata.pop('labels')
    nfeat = dg_graph.ndata.pop('features')
    if args.model == "GAT":
        graph = graph.add_self_loop()
    test_graph = None
    if (args.test_graph) != None:
        test_graph, _, num_classes = get_process_graph(args.test_graph, -1, testing = True)
        if args.model == "GAT":
            test_graph = test_graph.add_self_loop()

    ###################################
    #data = DglNodePropPredDataset(name=args.graph, root=root)
    #splitted_idx = data.get_idx_split()
    #train_idx, val_idx, test_idx = data.ndata['train_idx'], splitted_idx['valid'], splitted_idx['test']
    #graph, labels = data[0]
    #graph = graph.add_self_loop()
    #labels = labels[:, 0].to(device)

    # feat = graph.ndata.pop('feat')
    #year = graph.ndata.pop('year')

    in_feats = nfeat.shape[1]
    nfeat = nfeat.pin_memory()
    n_classes = (labels.max() + 1).item()
    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    metrics_queue = mp.Queue(4)
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, test_graph, metrics_queue

    #test_accs = []
    #for i in range(1, 11):
    #    print(f'\nRun {i:02d}:\n')
    #    test_acc = run(args, device, data)
    #    test_accs.append(test_acc)
    # test_accs = th.tensor(test_accs)
    print('============================')
    #print(f'Final Test: {test_accs.mean():.4f} ± {test_accs.std():.4f}')
    world_size = 4
    mp.spawn(
        run,
        args=(args,  data),
        nprocs=world_size,
        join=True
    )
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
    print("movement graph:{}".format(epoch_batch_graph))
    print("movement feature:{}".format(epoch_batch_feat_time))
    print("forward time:{}".format(epoch_batch_forward))
    print("data movement time:{}".format(epoch_batch_loadtime))
    print("backward time:{}".format(epoch_batch_backward))
