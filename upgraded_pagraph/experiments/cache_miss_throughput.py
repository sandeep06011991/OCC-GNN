
# Start Server
import subprocess
import os
import time
import sys

ROOT_DIR = "/home/spolisetty/OCC-GNN/upgraded_pagraph"
def start_server(filename):
    cmd = ['python3','{}/server/pa_server.py'.format(ROOT_DIR),'--dataset',filename]
    print(cmd)
    #cmd = ['python3','hello.py']
    fp = subprocess.Popen(cmd,
                    stdout = subprocess.PIPE,
                     stderr = subprocess.PIPE,
                     text = True)
    os.set_blocking(fp.stdout.fileno(), False)
    os.set_blocking(fp.stderr.fileno(),False)
    while(True):
        out = fp.stdout.readline()
        err = fp.stderr.readline()
        print(out,err)
        sys.stdout.flush()
        if 'start running graph' in str(out):
            print("Breaking")
            break
        time.sleep(1)
    print("Server is running can start client Finally")
    sys.stdout.flush()
    return fp
# Start client
import dgl
import torch
import time
import statistics
from PaGraph.storage.storage import GraphCacheServer
# Shared memory server has same data access bandwidth as local serverself.
def expected_throughput_and_latency(dataset):
    fp = start_server(dataset)
    dataname = dataset
    remote_g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
    num_nodes, num_edges = remote_g.proxy.get_graph_info(dataname)
    num_nodes, num_edges = int(num_nodes), int(num_edges)
    cache_percentage = 0
    device = 0

    storage = GraphCacheServer(remote_g, num_nodes, torch.arange(num_nodes), device , cache_percentage)
    avg_latency = []
    avg_bandwith_MBPS = []
    avg_local_gather = []
    batch_size = 4096
    num_runs = 20
    dummy = torch.rand(num_nodes, 100)
    for _ in range(num_runs):
        fetch = torch.randint(0,num_nodes,(batch_size,))
        t1 = time.time()
        data = storage.fetch_data(fetch)
        t2 = time.time()
        data = dummy[fetch].clone()
        t3 = time.time()
        avg_latency.append(t2-t1)
        avg_local_gather.append(t3-t2)
        avg_bandwith_MBPS.append((data.shape[0] * data.shape[1] * 4)/((t2-t1) * 1024 * 1024))
    print("Latency",  statistics.mean(avg_latency), statistics.variance(avg_latency))
    print("Bandwidth MBPS", statistics.mean(avg_bandwith_MBPS), statistics.variance(avg_bandwith_MBPS))
    print("Avg local gahter", statistics.mean(avg_local_gather))
    remote_g.destroy()
    fp.close()

if __name__=="__main__":
    expected_throughput_and_latency("ogbn-arxiv")
