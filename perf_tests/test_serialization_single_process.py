import set_path
from torch import multiprocessing as mp
import torch
from utils.utils import get_process_graph
from data.part_sample import Sample, Gpu_Local_Sample
from cslicer import cslicer
import time
import torch, statistics
import dgl
from utils.shared_mem_manager import *
from data.serialize import *
# Measure overhead without contention multiprocess contention.
# Absolute maximum.
def single_process_serialization_overhead(graph_name):
    batch_size = 4096
    dg_graph, partition_map, num_classes = get_process_graph(graph_name, -1)
    gpu_map = []
    for i in range(4):
        gpu_map.append(torch.where(partition_map == i)[0].tolist())
    fanout = 20
    deterministic = False
    testing = False
    self_edge = False
    slicer =  cslicer(graph_name, gpu_map, fanout, deterministic, testing, self_edge)
    orig_training_nodes = dg_graph.ndata["train_mask"].nonzero().flatten()
    print("training node size", orig_training_nodes)
    num_epochs = 4
    epoch_avg = []
    num_nodes = dg_graph.num_nodes()
    minibatch_sample = []
    avg_edges = []
    sm_filename_queue = mp.Queue(get_number_buckets(1))
    file_id = 1234
    sm_manager = SharedMemManager(sm_filename_queue, 1, file_id)
    worker_id = 0
    num_workers = 1
    queue = mp.Queue(2)
    device = 0
    sm_client = SharedMemClient(sm_filename_queue, "slicer", worker_id, num_workers,file_id)
    for i in range(num_epochs):
        j = 0
        training_nodes = orig_training_nodes[torch.randperm(orig_training_nodes.shape[0])]
        t1 = time.time()
        # if i == 1:
        #     minibatch_sample = []
        while j < training_nodes.shape[0]:
            t11 = time.time()
            if j + batch_size > training_nodes.shape[0]:
                break
            csample = slicer.getSample(training_nodes[j:j+ batch_size].tolist())
            tensorized_sample = Sample(csample)
            gpu_local_samples = []
            for gpu_id in range(1):
                # gpu_local_samples.append(Gpu_Local_Sample(tensorized_sample, gpu_id))
                obj = Gpu_Local_Sample()
                obj.set_from_global_sample(tensorized_sample,gpu_id)
                data = serialize_to_tensor(obj)
                data = data.numpy()
                name = sm_client.write_to_shared_memory(data)
                ref = ((name, data.shape, data.dtype.name))
                gpu_local_samples.append(ref)
            queue.put(gpu_local_samples[0])
            meta  = queue.get()
            (name, shape, dtype ) = meta
            dtype = np.dtype( dtype )
            t3 = time.time()
            tensor = sm_client.read_from_shared_memory(name, shape, dtype)
            t4 = time.time()
            tensor = tensor.to(device)
            # tensor = tensor.long()
            gpu_local_sample = Gpu_Local_Sample()
            device = torch.device(device)
            # Refactor this must not be moving to GPU at this point.
            construct_from_tensor_on_gpu(tensor, device, gpu_local_sample)
            gpu_local_sample.prepare()
            t5 = time.time()
            # Memory can be released now as object is constructed from tensor.
            sm_client.free_used_shared_memory(name)

            j = j + batch_size
            t22 = time.time()
            avg_edges.append(tensorized_sample.get_number_of_edges())
            minibatch_sample.append(t22-t11)
        t2 = time.time()
        epoch_avg.append(t2-t1)
    print(epoch_avg)
    print(minibatch_sample)
    # assert(statistics.variance(epoch_avg) < .10 * statistics.mean(epoch_avg))
    # assert(statistics.variance(minibatch_sample) < .1 * statistics.mean(minibatch_sample))
    return {"minibatch": statistics.mean(minibatch_sample), "epoch": statistics.mean(epoch_avg),\
        "edges": statistics.mean(avg_edges)}

def baseline_single_process_throughput(graph):
    fanout = 20
    no_layers = 3
    batch_size = 4096
    num_workers = 1
    graph, partition_map, num_classes = get_process_graph(graph, -1)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for _ in range(no_layers)], replace = False)
    train_nid = graph.ndata['train_mask'].nonzero()
    train_nid = train_nid.flatten()
    dataloader_i = dgl.dataloading.NodeDataLoader(
        graph,
        train_nid,
        sampler,
        device='cpu',
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)
    avg_epoch = []
    avg_sample = []
    avg_fanout = []
    avg_edges = []
    avg_degree = []
    avg  = []
    for i in range(4):
        dataloader = iter(dataloader_i)
        t11 = time.time()
        if i ==1:
            avg_sample = []
        while(True):
            try:
                t1 = time.time()
                input_nodes, seeds, blocks = next(dataloader)
                t2 = time.time()
                avg.append(t2-t1)
                s = 0
                for b in blocks:
                    s = s + b.number_of_edges()
                avg_edges.append(s)
                avg_fanout.append(input_nodes.shape[0]/seeds.shape[0])
            except StopIteration:
                break
        # print("avg sample",avg)
        t22 = time.time()
        assert(statistics.variance(avg[1:])/statistics.mean(avg[1:]) < .1)
        avg_sample.append(sum(avg[1:])/len(avg[1:]))
        avg_epoch.append(t22-t11)
        # print("sample", avg)
        assert(statistics.variance(avg_fanout[1:])/statistics.mean(avg_fanout[1:])<.1)
        # print("avg fanout", avg_fanout)
        # print(statistics.variance(avg_epoch[1:]) / statistics.mean(avg_epoch[1:]))
    # print(avg_epoch)
    # print(statistics.variance(avg_epoch[1:]) / statistics.mean(avg_epoch[1:]))
    assert(statistics.variance(avg_epoch[1:]) / statistics.mean(avg_epoch[1:]) < .1)
    assert(statistics.variance(avg_sample[1:])/ statistics.mean(avg_epoch[1:]) < .10)
    minibatch_sample = sum(avg_sample[1:])/len(avg_sample[1:])
    epoch_sample = sum(avg_epoch[1:])/len(avg_epoch[1:])

    return {"minibatch": minibatch_sample, "epoch": epoch_sample}

if __name__ == "__main__":
    graph_name = "ogbn-products"
    # print(single_process_serialization_overhead(graph_name))
    print(baseline_single_process_throughput(graph_name))
