import torch
import dgl
import os
from env import *
import time 

# Not using logging as I dont need to persist the time taken to preprocess.
# When that need arises move all prints to logging 
ROOT_DIR = get_data_dir()
def write_storage_order_book(graph_name:str, cache_size:str):
    TARGET_DIR = ROOT_DIR + "/" + graph_name
    with open(TARGET_DIR + "/meta.txt",'r') as fp:
        lines = fp.readlines()
    meta = {}
    for line in lines:
        k, v = line.split("=")
        print(k,v)
        meta[k] = int(v)

    with open(TARGET_DIR + "/partition_offsets.txt", 'r') as fp:
        lines = fp.readlines()
    partition_offsets = lines[0].split(",")    
    partition_offsets = [int(i) for i in partition_offsets]

    num_nodes = meta['num_nodes']
    if cache_size[-2:] == "GB":
        size_in_bytes = (1024 ** 3) * int(cache_size[:-2])
    if cache_size[-2:] == "MB":
        size_in_bytes = (1024 ** 2) * int(cache_size[:-2])

    percentage_of_nodes_to_cache = size_in_bytes/(meta['num_nodes'] * meta['feature_dim'] *  4)
    percentage_of_nodes_to_cache = min(1, percentage_of_nodes_to_cache)
    nodes_to_cache = int(percentage_of_nodes_to_cache * num_nodes)

    print(nodes_to_cache, "nodes to cache")
    fsize = meta['feature_dim']
    import numpy as np
    num_partitions = 4
    ordered_partition_nodes = []
    for partition in range(num_partitions):
        ordered_partition_nodes.append(\
                (torch.arange(partition_offsets[partition],partition_offsets[partition + 1])))    
    out_degree = torch.from_numpy(np.fromfile(TARGET_DIR + "/out_degrees.bin",dtype = np.int32))

    orderbook = []
    for curr_partition in range(num_partitions):
        global_nodes_not_in_partition = []
        local_ordering = []
        nodes_to_cache_for_partition = nodes_to_cache
        for i in range(num_partitions):
            if (i == curr_partition):
                global_nodes_in_partition = ordered_partition_nodes[i]
            else:
                global_nodes_not_in_partition.append(ordered_partition_nodes[i])
            local_ordering.append(partition_offsets[i])
        num_nodes_from_self =  min(nodes_to_cache, global_nodes_in_partition.shape[0])
        nodes_to_cache_for_partition -= num_nodes_from_self
        local_ordering[curr_partition] = partition_offsets[curr_partition] + \
                                                        num_nodes_from_self
        if(nodes_to_cache_for_partition == 0):
            orderbook.append(local_ordering)
            continue
        global_nodes_not_in_partition = torch.cat(global_nodes_not_in_partition)
        not_in_partition_degree = out_degree[global_nodes_not_in_partition]
        values, indices = torch.sort(not_in_partition_degree, descending = True)
        global_nodes_not_in_partition_cached =global_nodes_not_in_partition[indices[:nodes_to_cache_for_partition]]
        for i in range(num_partitions):
            if i == curr_partition:
                continue
            selected =  (global_nodes_not_in_partition_cached >= partition_offsets[i] ) \
                & (global_nodes_not_in_partition_cached < partition_offsets[i + 1])
            local_ordering[i] = partition_offsets[i] + global_nodes_not_in_partition_cached[selected].shape[0]   
        orderbook.append(local_ordering)
    with open(f"{TARGET_DIR}/order_book_{cache_size}.txt", 'w') as fp:
        for p_offsets in orderbook:
            str_offsets = [str(p) for p in p_offsets]
            str_offsets = ",".join(str_offsets)
            fp.write(f"{str_offsets}\n")    

if __name__ == "__main__":
    write_storage_order_book("ogbn-products", "2GB")
    #write_storage_order_book("ogbn-papers100M", "2GB")    
