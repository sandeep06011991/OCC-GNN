import torch
import dgl
import time
import nvtx
from models.dist_gcn import get_sage_distributed
from models.dist_gat_v2 import get_gat_distributed
from utils.utils import get_process_graph
from utils.memory_manager import MemoryManager, GpuLocalStorage
import torch.optim as optim
from data import Bipartite, Sample, Gpu_Local_Sample
import numpy as np
import torch.multiprocessing as mp
import random
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import os
import time
import inspect
from utils import utils
from utils.utils import *
from cu_shared import *
from data.serialize import *
import logging
from test_accuracy import *
from layers.shuffle_functional import *
from measure import *


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum(),len(pred )

def train_minibatch(target_nodes, num_gpus, partition_offsets,\
                    sampler, args, exchange_queue, optimizer, gpu_local_storage,\
                        attention, labels, events, isTrain, loss_fn, proc_id, model, val_acc_queue):
    minibatch_metrics = MinibatchMetrics()
    for i in range(num_gpus):
        print(partition_offsets[i], partition_offsets[i+1], target_nodes[:10])
        print((target_nodes >= partition_offsets[i])\
                            & (target_nodes < partition_offsets[i+1]))
        n = torch.where((target_nodes >= partition_offsets[i])\
                            & (target_nodes < partition_offsets[i+1]))[0].shape[0]
        print("load map ", n, proc_id)
    torch.cuda.nvtx.range_push("Minibatch")
    sample_get_start = time.time()
    send_dict = {}
    recv_dict = {}
    my_gpu_local_sample = None
    print("Memory allocated ", torch.cuda.memory_allocated())
    csample = sampler.getSample(target_nodes.tolist(), args.load_balance)
    tensorized_sample = Sample(csample)
    sample_id = tensorized_sample.randid
    for gpu_id in range(num_gpus):
        obj = Gpu_Local_Sample()
        obj.set_from_global_sample(tensorized_sample,gpu_id)
        if gpu_id == proc_id:
            my_gpu_local_sample = obj
            continue   
    #print("Temporary fix serializing on cpu")
        if False:
            # CPU sampler is never used
            data = serialize_to_tensor(obj, torch.device('cpu'), num_gpus = gpus)
        else:
            data = serialize_to_tensor(obj, torch.device(proc_id), num_gpus = num_gpus)
            if gpu_id != proc_id:
                send_dict[gpu_id] = data.clone()
    
        sample_id = proc_id
        for_worker_id = gpu_id
        #name = sm_client.write_to_shared_memory(data, for_worker_id, sample_id)
        #ref = ((sample_id, for_worker_id, name, data.shape, data.dtype.name))
        ref = (sample_id, for_worker_id, data.shape)
        assert(len(data.shape) != 0)
        exchange_queue[for_worker_id].put(ref)
    
    print("Memory allocated sampling ", torch.cuda.memory_allocated())    
    del csample
        # gc.collect()
        # torch.cuda.empty_cache()
        
    sample_get_end = time.time()
    minibatch_metrics.sample_get_time =  sample_get_end - sample_get_start
        # exchange
    t1 = time.time()
    for gpu_id in range(num_gpus):
            # Read my own exchange_queue
        if(gpu_id) == proc_id:
            recv_dict[gpu_id] = torch.empty([0], device = proc_id, dtype = torch.int32)
            continue
            
        sample_id, for_worker_id, shape = exchange_queue[proc_id].get()
        assert(for_worker_id == proc_id)
        if shape != "EMPTY":
            print(sample_id, shape, "SHAPE TO SHUFFLE")
            recv_dict[sample_id] = torch.empty(shape, device = proc_id, dtype = torch.int32)
        else:
            recv_dict[sample_id] = torch.empty([0], device= proc_id, dtype = torch.int32)

    print("Memory allocated shuffling", torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    print([send_dict[i].shape for i in send_dict.keys()],\
            [recv_dict[i].shape for i in recv_dict.keys()], "Shuffling ", proc_id)
    shuffle_functional(proc_id, send_dict, recv_dict, num_gpus)
    torch.cuda.synchronize()
    t2 = time.time()
    del send_dict

    minibatch_metrics.movement_graph = (t2 - t1)
    minibatch_metrics.movement_feat = 0
    minibatch_metrics.data_moved_per_gpu = 0
    minibatch_metrics.data_moved_inter_gpu = 0
    minibatch_metrics.edges_per_gpu = 0
    minibatch_metrics.forward_time = 0
    minibatch_metrics.backward_time = 0
    for i in range(num_gpus):
        optimizer.zero_grad()
        t1 = time.time()
        if i == proc_id:
            gpu_local_sample = my_gpu_local_sample
            if type(gpu_local_sample) is type(torch.tensor([])):
                continue
        else:
            tensor = recv_dict[i]
            if(tensor.shape[0] == 0):
                continue
            gpu_local_sample = Gpu_Local_Sample()
            device = torch.device(proc_id)
            #print(tensor.shape, "RECOEVECD", tensor.dtype, torch.sum(tensor))
            construct_from_tensor_on_gpu(tensor, device, gpu_local_sample, num_gpus = num_gpus)
        # construct_from_tensor_on_gpu(tensor, device, gpu_local_sample, num_gpus = gpus)
        # gpu_local_sample.debug()
        torch.cuda.nvtx.range_push("prepare")
        gpu_local_sample.prepare(attention)
        torch.cuda.nvtx.range_pop()
        t2 = time.time()
        minibatch_metrics.movement_graph += (t2 - t1)
        classes = labels[gpu_local_sample.out_nodes].to(torch.device(proc_id))
        torch.cuda.set_device(proc_id)
        #optimizer.zero_grad()
        m_t1 = time.time()
        torch.cuda.nvtx.range_push("storage")
        input_features  = gpu_local_storage.get_input_features(gpu_local_sample.cache_hit_from, \
                gpu_local_sample.cache_hit_to, gpu_local_sample.cache_miss_from, gpu_local_sample.cache_miss_to)
        minibatch_metrics.movement_feat = time.time() - m_t1
        torch.cuda.nvtx.range_pop()
        edges, nodes, edge_split, node_split = gpu_local_sample.get_edges_and_send_data()
        minibatch_metrics.edges_per_gpu += edges
        minibatch_metrics.data_moved_per_gpu += ((gpu_local_sample.cache_miss_from.shape[0] * \
                                                    gpu_local_storage.features.shape[1] * 4)/(1024 * 1024)) +\
                            ((nodes * args.num_hidden * 4)/(1024 * 1024))
        minibatch_metrics.data_moved_inter_gpu  += ((nodes * args.num_hidden *4)/(1024 * 1024))
        events[0].record()
        torch.cuda.nvtx.range_push("training {}:{}".format(edge_split, node_split))
        output = model.forward(gpu_local_sample, input_features, None)
        events[1].record()
        if isTrain:
            loss = loss_fn(output,classes)/args.batch_size
            loss.backward()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            events[2].record()
            optimizer.step()
        else:
            acc = compute_acc(output,classes)
            correct, total = (acc[0].item(),acc[1])

            if(proc_id == 0):
                for _ in range(num_gpus - 1):
                    correct_,total_ = val_acc_queue.get()
                    print(acc, correct_, total_, "Popped ")
                    correct = correct + correct_ 
                    total = total + total_
                minibatch_metrics.correct = correct
                minibatch_metrics.predicted = total 
            else: 
                val_acc_queue.put((acc[0].item(), acc[1]))    
        minibatch_metrics.forward_time += events[0].elapsed_time(events[1])/1000
        if isTrain:
            minibatch_metrics.backward_time += events[1].elapsed_time(events[2])/1000
        # print("Training time", fp_start.elapsed_time(fp_end)/1000, fp_end.elapsed_time(bp_end)/1000)
        #assert(False)
    return minibatch_metrics

def run_trainer_process(proc_id,  num_gpus, features, args\
                        ,num_classes,  labels,\
                           cache_percentage,  \
                             fanout, exchange_queue,\
                            graph_name, num_layers, train_nid, valid_nid, val_acc_queue):
    print("Trainer process starts!!!, ",proc_id)
    torch.cuda.set_device(proc_id)
    from utils.utils import get_order_book
    order_book = get_order_book(graph_name, cache_percentage)
    partition_offsets = get_partition_offsets(graph_name)
    gpu_local_storage  = GpuLocalStorage(cache_percentage, features, order_book, partition_offsets, proc_id)
    if proc_id == 0:
        ## Configure logger
        os.makedirs('{}/logs'.format(PATH_DIR),exist_ok = True)
        FILENAME= ('{}/logs/{}_{}_{}_{}.txt'.format(PATH_DIR, \
                 args.graph, args.batch_size, cache_percentage ,args.model))
        fileh = logging.FileHandler(FILENAME, 'w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileh.setFormatter(formatter)
        flog = logging.getLogger()  # root logger
        flog.addHandler(fileh)      # set the new handler
        flog.setLevel(logging.INFO)

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = num_gpus
    assert(world_size > 0)

    torch.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    deterministic = False
    if args.model == "gcn":
        model = get_sage_distributed(args.num_hidden, features, num_classes,
            proc_id, deterministic, args.model, num_gpus,\
                    args.num_layers, args.skip_shuffle)
        self_edge = False
        attention = False
        pull_optimization = False
    else:   
        assert(args.model == "gat" or args.model == "gat-pull")
        if(args.model == "gat"):
            pull_optimization = False
        else:
            pull_optimization = True
        
        model = get_gat_distributed(args.num_hidden, features, num_classes,
                proc_id, deterministic, args.model, pull_optimization, num_gpus,  args.num_layers, args.skip_shuffle)
        self_edge = True
        attention = True
    
    rounds = 3
    deterministic = False
    testing = False
    use_cpu_sampler = False
    if use_cpu_sampler:
        print("Using CPU Slicer")
        from cslicer import cslicer
        # sampler = cslicer(graph_name, storage_vector, fanout[0],\
        #             deterministic, testing , self_edge, rounds, \
        #             pull_optimization, num_layers, num_gpus)
    else:
        print("using GPU Sampler", args.use_uva)
        from cuslicer import cuslicer
        sampler = cuslicer(graph_name, cache_percentage,
                fanout ,deterministic, testing, self_edge, rounds, pull_optimization, num_layers, num_gpus, proc_id, args.random_partition, args.use_uva)
    
    device = proc_id
    if proc_id ==0:
        print(args.test_graph_dir)
        if args.test_graph_dir != None:
            device = 0
            test_acc_func = TestAccuracy(args.test_graph_dir, device, self_edge)
        else:
            test_acc_func = None
    
    model = model.to(proc_id)
    model =  DistributedDataParallel(model, device_ids = [proc_id],\
                output_device = proc_id)
                # find_unused_parameters = False
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #in_degrees = in_degrees.to(proc_id)
    torch.cuda.set_device(proc_id)
    # print("Features ", features.device)
    num_epochs = args.num_epochs
    
    global_order_dict[Bipartite] = get_attr_order_and_offset_size(Bipartite(), num_partitions = num_gpus)
    global_order_dict[Gpu_Local_Sample] = get_attr_order_and_offset_size(Gpu_Local_Sample(), num_partitions = num_gpus)
    labels = labels.to(proc_id)
    print("Training nodes",train_nid.shape)
    # Test splits for equality
    check_splits = train_nid.split(train_nid.size(0)//world_size)
    for i in range(1,4):
        assert(check_splits[0].shape[0]// args.batch_size == check_splits[i].shape[0]// args.batch_size)
    check_splits = valid_nid.split(valid_nid.size(0)//world_size)
    for i in range(1,4):
        assert(check_splits[0].shape[0]// args.batch_size == check_splits[i].shape[0]// args.batch_size)


    train_nid = train_nid.split(train_nid.size(0) // world_size)[proc_id]
    valid_nid = valid_nid.split(valid_nid.size(0) // world_size)[proc_id]

    print("Training epocs", num_epochs)
    experiment_metrics = ExperimentMetrics() 
    events = [torch.cuda.Event(enable_timing= True) for i in range(4)]
    for epoch_no in range(num_epochs):
        random.shuffle(train_nid)
        num_minibatches = train_nid.size(0) // args.batch_size
        print("num minibatches", num_minibatches)
        epoch_metrics = EpochMetrics()
        t1 = time.time()
        for minibatch in range(num_minibatches):
            batch_nodes = train_nid[minibatch * args.batch_size : (minibatch + 1) * args.batch_size]
            isTrain = True 
            minibatch = train_minibatch(batch_nodes, num_gpus, partition_offsets,\
                    sampler, args, exchange_queue, optimizer, gpu_local_storage,\
                        attention, labels, events, isTrain, loss_fn, proc_id, model,val_acc_queue)
            epoch_metrics.append(minibatch)
        t2 = time.time()    
        epoch_time = t2 - t1
        num_val_minibatches = valid_nid.size(0)//args.batch_size
        correct = 0
        total = 0 
        for minibatch in range(num_val_minibatches):
            batch_nodes = valid_nid[minibatch * args.batch_size : (minibatch + 1) * args.batch_size]
            isTrain = False 
            minibatch_metrics = train_minibatch(batch_nodes, num_gpus, partition_offsets,\
                    sampler, args, exchange_queue, optimizer, gpu_local_storage,\
                        attention, labels, events, isTrain, loss_fn, proc_id, model, val_acc_queue)
            if proc_id == 0:
                correct += minibatch_metrics.correct
                total += minibatch_metrics.predicted
            else:
                correct = 0
                total = 1
        experiment_metrics.append(epoch_metrics, correct/total, epoch_time)
    
    if proc_id == 0:
        print(experiment_metrics)
