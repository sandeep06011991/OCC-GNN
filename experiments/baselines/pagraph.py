 # Ensure cuda/10.2 is loaded
import subprocess
import os
import time
from utils.utils import *
# ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/upgraded_pagraph"
# DATA_DIR = "/home/spolisetty_umass_edu/OCC-GNN/experiments/exp6/"
# PA_ROOT_DIR = "/home/spolisetty/OCC-GNN/upgraded_pagraph"
# DATA_DIR = "/home/spolisetty/OCC-GNN/experiments/exp6/"
import sys
import re
import git

from normalize import *

Feat = {"ogbn-arxiv":"128","ogbn-products":"100", "reorder-papers100M":"128","amazon":200}
ROOT_DIR = PA_ROOT_DIR
def get_git_info():
    repo = git.Repo(search_parent_directories = True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()
    return sha,dirty

def check_path():
    path_set = False
    for p in sys.path:
        print(p)
        if ROOT_DIR ==  p:
            path_set = True
    if (not path_set):
        print("Setting Path")
        sys.path.append(ROOT_DIR)
def avg(ls):
    return (sum(ls[1:]/(len(ls)-1)))

def check_no_stale():
    ps = os.popen('ps').read().split('\n')
    py_process = 0
    for p in ps:
        if 'python3' in p:
            py_process += 1
    if py_process != 1:
        print("stale processes exist clean up")
    assert(py_process == 1)
    # I can add a kill ad clean up later.


# Start server and wait for ready
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

def parse_float(string, output):
    matches = re.findall(string,output)
    assert(len(matches) == 1)
    return float(matches[0])

def start_client(filename, model, num_hidden, batch_size, cache_per, \
            fanout, num_layers, sample_gpu):
    feat_size = Feat[filename]
    print(cache_per)
    cmd = ['python3','{}/examples/profile/pa_gcn.py'.format(ROOT_DIR), '--dataset', filename,'--n-epochs','5'\
                    , '--n-hidden', str(num_hidden), '--batch-size', str(batch_size),\
                        '--model', model, "--cache-per",str(cache_per), \
                            '--fan-out', fanout, '--n-layers', str(num_layers)]
    if sample_gpu:
        cmd.append('--sample-gpu')
    output = subprocess.run(cmd, capture_output=True)
    out = str(output.stdout)
    err = str(output.stderr)
    print(out,err)
    output = out
    accuracy = parse_float("accuracy: (\d+\.\d+)", output)
    move_graph = parse_float("movement graph:(\d+\.\d+)", output)
    move_feat = parse_float("movement feature:(\d+\.\d+)", output)
    total_movement = parse_float("data movement:(\d+\.\d+)",output)
    forward = parse_float("forward time:(\d+\.\d+)", output)
    
    backward = parse_float("backward time:-?(\d+\.\d+)", output)
    sample = parse_float("sample_time:(\d+\.\d+)",output)
    #compute  = float(re.findall("Compute time: (\d+\.\d+)s",output)[0])
    # collect  = float(re.findall("CPU collect: (\d+\.\d+)s",output)[0])
    # move  = float(re.findall("CUDA move: (\d+\.\d+)s",output)[0])
    epoch  = parse_float("Epoch time: (\d+\.\d+)s",output)
    miss_rate = parse_float("Miss rate: (\d+\.\d+)s",output)
    miss_num = re.findall("Miss num per epoch: (\d+\.\d+)MB, device \d+",output)
    s = 0
    e =0
    edges_processed = re.findall("Edges processed per epoch: (\d+\.\d+)",output)
    for i in range(4):
        s = s + int(float(miss_num[i]))
        e = e + int(float(edges_processed[i]))
    miss_num = s/4
    edges_processed = e/4
    #miss_num = int(float(re.findall("Miss num per epoch: (\d+\.\d+)MB, device \d+",output)[0]))
    sample, total_movement, forward, backward = normalize(epoch, sample, total_movement, forward, backward)

    #edges_processed = int(float(re.findall("Edges processed per epoch: (\d+\.\d+)",output)[0]))
    return {"sample":sample, "forward": forward, "backward": backward,\
            "move_feat":"{:.3f}".format(move_feat), "epoch_time":epoch, "miss_rate": miss_rate, "move_graph":move_graph\
            , "accuracy": accuracy, "miss_num":miss_num, "edges":edges_processed, "total_movement":total_movement}

def run_experiment_on_graph(filename, model, hidden_size, batch_size, cache_per,\
            fanout, num_layers, sample_gpu):
    fp = start_server(filename)
    feat_size = Feat[filename]

    res = start_client(filename, model, hidden_size, batch_size, cache_per, \
                fanout, num_layers, sample_gpu)
    fp.terminate()
    return res
    # WRITE = "{}/exp6_{}_pagraph.txt".format(OUT_DIR, SYSTEM)
    # with open(WRITE,'a') as fp:
    #     fp.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(filename, "jupiter", cache_per, hidden_size, feat_size, \
    #             4 * batch_size, model, res["sample"], res["total_movement"],res["forward"], \
    #                 res["backward"],res["epoch_time"], res["accuracy"],res["miss_num"], res["edges"] ))
    # fp.close()
