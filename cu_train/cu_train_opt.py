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


def avg(ls):
    # assert(len(ls) > 3)
    print(ls)
    if len(ls) == 1:
        return ls[0]
    if(len(ls) <= 3):
        return sum(ls[1:])/len(ls[1:])
    a = max(ls[1:])
    b = min(ls[1:])
    # remove 3 as (remove first, max and min)
    return (sum(ls[1:]) - a - b)/(len(ls) - 3)

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    false_labels = torch.where(torch.argmax(pred,dim = 1) != labels)[0]
    return (torch.argmax(pred, dim=1) == labels).float().sum(),len(pred )


def run_trainer_process(proc_id, gpus, sample_queue,  minibatches_per_epoch, features, args\
                        ,num_classes,  labels,\
                             deterministic,\
                           cache_percentage,  epochs_required,\
                             fanout, exchange_queue,\
                            graph_name, num_layers, num_gpus, shbuffs, mpbarrier):
    print("Trainer process starts!!!, ",proc_id)
    torch.cuda.set_device(proc_id)
    from utils.utils import get_process_graph
    from utils.utils import get_order_book
    dg_graph,partition_offsets, num_classes = get_process_graph(graph_name, -1 , 4)
    order_book = get_order_book(graph_name, cache_percentage)
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
    world_size = gpus
    assert(world_size > 0)

    torch.distributed.init_process_group(backend="nccl",\
             init_method=dist_init_method,  world_size=world_size,rank=proc_id)
    # print("SEED",torch.seed())
    current_gpu = proc_id
    if args.model == "gcn":
        model = get_sage_distributed(args.num_hidden, features, num_classes,
            proc_id, args.deterministic, args.model, gpus,\
                    args.num_layers, args.skip_shuffle, args.barrier, shbuffs, mpbarrier)
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
                proc_id, args.deterministic, args.model, pull_optimization, gpus,  args.num_layers, args.skip_shuffle)
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

    sample_get_epoch = []
    forward_epoch = []
    backward_epoch = []
    movement_graph_epoch = []
    movement_feat_epoch = []
    epoch_time = []
    epoch_accuracy = []
    data_moved_per_gpu_epoch =[]
    data_moved_inter_gpu_epoch = []
    edges_per_gpu_epoch = []
    first_layer_time_epoch = []
    other_layer_time_epoch = []
    sample_get_time = 0
    first_layer_time = 0
    other_layer_time = 0
    forward_time = 0
    backward_time = 0
    movement_graph = 0
    movement_feat = 0
    data_moved_per_gpu = 0
    data_moved_inter_gpu = 0
    edges_per_gpu = 0
    movement_partials_epoch = []
    #in_degrees = in_degrees.to(proc_id)
    fp_start = torch.cuda.Event(enable_timing=True)
    fp_end = torch.cuda.Event(enable_timing=True)
    bp_end = torch.cuda.Event(enable_timing=True)
    accuracy = 0
    torch.cuda.set_device(proc_id)
    test_accuracy_list = []
    current_minibatch = 0
    # print("Features ", features.device)
    num_epochs = 0
    minibatch_sample_time = []
    if proc_id == 0:
        flog = logging.getLogger()
    e_t1 = time.time()
    global_order_dict[Bipartite] = get_attr_order_and_offset_size(Bipartite(), num_partitions = gpus)
    global_order_dict[Gpu_Local_Sample] = get_attr_order_and_offset_size(Gpu_Local_Sample(), num_partitions = gpus)
    labels = labels.to(proc_id)
    while(True):
        print("try to get sample", current_minibatch , minibatches_per_epoch, num_epochs, epochs_required)
        if num_epochs == epochs_required :
            break
        training_node= sample_queue.get()
        torch.cuda.nvtx.range_push("Minibatch")
        sample_get_start = time.time()
        send_dict = {}
        recv_dict = {}
        my_gpu_local_sample = None
        if(type(training_node) != type("")):
            
            # args.load_balance   
            print("Memory allocated ", torch.cuda.memory_allocated())
            
            csample = sampler.getSample(training_node, args.load_balance)
            tensorized_sample = Sample(csample)
            
            sample_id = tensorized_sample.randid
            for gpu_id in range(num_gpus):
                obj = Gpu_Local_Sample()
                obj.set_from_global_sample(tensorized_sample,gpu_id)
                if gpu_id == proc_id:
                    my_gpu_local_sample = obj
                    continue   
                #print("Temporary fix serializing on cpu")
                if use_cpu_sampler:
                    assert(False)
                    data = serialize_to_tensor(obj, torch.device('cpu'), num_gpus = gpus)
                else:
                    data = serialize_to_tensor(obj, torch.device(proc_id), num_gpus = gpus)
                    if gpu_id != proc_id:
                        send_dict[gpu_id] = data.clone()
                #data = data.to('cpu').numpy()
                #print("Warning shared memory queue may be  emtpy ", sm_filename_queue.qsize())
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
        else:
            for gpu_id in range(num_gpus):
                if gpu_id == proc_id:
                   my_gpu_local_sample = torch.tensor([], device = proc_id,dtype = torch.int32)
                   continue
                sample_id = proc_id
                for_worker_id = gpu_id
                send_dict[gpu_id] = torch.empty([0], device = proc_id, dtype = torch.int32)
                ref = (sample_id, for_worker_id, "EMPTY")
                exchange_queue[for_worker_id].put(ref)
        
    
        sample_get_end = time.time()
        sample_get_time += sample_get_end - sample_get_start
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
        shuffle_functional(proc_id, send_dict, recv_dict, gpus)
        torch.cuda.synchronize()
        t2 = time.time()
        del send_dict

        movement_graph += (t2 - t1)
        
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
                device = torch.device(device)
                #print(tensor.shape, "RECOEVECD", tensor.dtype, torch.sum(tensor))
                construct_from_tensor_on_gpu(tensor, device, gpu_local_sample, num_gpus = gpus)
            #construct_from_tensor_on_gpu(tensor, device, gpu_local_sample, num_gpus = gpus)
            # gpu_local_sample.debug()
            # FixME: What is attention ?
            torch.cuda.nvtx.range_push("prepare")
            gpu_local_sample.prepare(attention)
            torch.cuda.nvtx.range_pop()
#             if proc_id == 0:
#                 if test_acc_func != None:
#                     test_accuracy = test_acc_func.get_accuracy(model,flog)
#                     test_accuracy_list.append(test_accuracy)
#                     print("test_accuracy_log:{}, epoch:{}".format(test_accuracy, num_epochs-1))

#         #assert(features.device == torch.device('cpu'))
#         #gpu_local_sample.debug()
            t2 = time.time()
            movement_graph += (t2 - t1)
            classes = labels[gpu_local_sample.out_nodes].to(torch.device(proc_id))
#         # print("Last layer nodes",gpu_local_sample.last_layer_nodes)
            torch.cuda.set_device(proc_id)
            #optimizer.zero_grad()
            m_t1 = time.time()
            torch.cuda.nvtx.range_push("storage")
            input_features  = gpu_local_storage.get_input_features(gpu_local_sample.cache_hit_from, \
                    gpu_local_sample.cache_hit_to, gpu_local_sample.cache_miss_from, gpu_local_sample.cache_miss_to)
            torch.distributed.barrier()
            movement_feat += time.time() - m_t1
            torch.cuda.nvtx.range_pop()
            edges, nodes, edge_split, node_split = gpu_local_sample.get_edges_and_send_data()
            edges_per_gpu += edges
            data_moved_per_gpu += ((gpu_local_sample.cache_miss_from.shape[0] * features.shape[1] * 4)/(1024 * 1024)) +\
                                ((nodes * args.num_hidden * 4)/(1024 * 1024))
            data_moved_inter_gpu  += ((nodes * args.num_hidden *4)/(1024 * 1024))
            fp_start.record()
            torch.cuda.nvtx.range_push("training {}:{}".format(edge_split, node_split))
            output = model.forward(gpu_local_sample, input_features, None)
            loss = loss_fn(output,classes)/args.batch_size
            fp_end.record()
            loss.backward()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            bp_end.record()

            optimizer.step()
            if(current_minibatch == 10):
                torch.cuda.profiler.start()
            if(current_minibatch == 30):
                torch.cuda.profiler.stop()

            first_layer, other_layer = model.module.get_reset_layer_time()
            first_layer_time += first_layer
            other_layer_time += other_layer
            # print(proc_id, "Does both forward and backward!!", current_minibatch)
            current_minibatch += 1
            if (current_minibatch == minibatches_per_epoch):
                sample_get_epoch.append(sample_get_time)
                forward_epoch.append(forward_time)
                backward_epoch.append(backward_time)
                movement_graph_epoch.append(movement_graph)
                movement_feat_epoch.append(movement_feat)
                edges_per_gpu_epoch.append(edges_per_gpu)
                data_moved_per_gpu_epoch.append(data_moved_per_gpu)
                data_moved_inter_gpu_epoch.append(data_moved_inter_gpu)
                movement_partials_epoch.append(model.module.get_reset_shuffle_time())
                first_layer_time_epoch.append(first_layer_time)
                other_layer_time_epoch.append(other_layer_time)
                e_t2 = time.time()
                epoch_time.append(e_t2-e_t1)
                first_layer_time = 0
                other_layer_time = 0
                sample_get_time = 0
                forward_time = 0
                backward_time = 0
                movement_graph = 0
                movement_feat = 0
                edges_per_gpu = 0
                data_moved_per_gpu = 0
                data_moved_inter_gpu = 0
                num_epochs += 1
                e_t1 = time.time()
                current_minibatch = 0

                if args.test_graph_dir != None and proc_id == 0 :
                    test_accuracy =  test_acc_func.get_accuracy( model )
                    test_accuracy_list.append(test_accuracy)

            if output.shape[0] !=0:
                acc = compute_acc(output,classes)
                acc = (acc[0].item()/acc[1])
                print("Accuracy ", acc, current_minibatch ,num_epochs)
            torch.cuda.synchronize(bp_end)
            torch.cuda.nvtx.range_pop()
            forward_time += fp_start.elapsed_time(fp_end)/1000
            backward_time += fp_end.elapsed_time(bp_end)/1000
            # print("Training time", fp_start.elapsed_time(fp_end)/1000, fp_end.elapsed_time(bp_end)/1000)
            #assert(False)
    print("Exiting main training loop",sample_queue.qsize())
#     dev_id = proc_id
    #if proc_id == 1:
#     #     prof.dump_stats('worker.lprof')
    print("edges per epoch:{}".format(avg(edges_per_gpu_epoch)))
    if proc_id == 0:
        print("Test Accuracy:", test_accuracy_list)
        print("accuracy:{}".format(acc))
        print("#################",epoch_time)
        print("epoch_time:{}".format(avg(epoch_time)))
        print("sample_time:{}".format(avg(sample_get_epoch)))
        print("movement graph:{}".format(avg(movement_graph_epoch)))
        print("movement feature:{}".format(avg(movement_feat_epoch)))
        print("forward time:{}".format(avg(forward_epoch)))
        print("backward time:{}".format(avg(backward_epoch)))
        print("data movement:{}MB".format(avg(data_moved_per_gpu_epoch)))
        #print("Inter gpu data movement:{}MB".format(avg(data_moved_inter_gpu_epoch)))
        print("Shuffle time:{}".format(avg(movement_partials_epoch)))
        print("Memory Used:{} GB".format(torch.cuda.max_memory_allocated()/(1024 ** 3)))
        print("First layer time:{}".format(avg(first_layer_time_epoch)))
        print("other layer time:{}".format(avg(other_layer_time_epoch)))
#     print("Trainer returns")
#         # print("Memory",torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
     # print("Thread running")
