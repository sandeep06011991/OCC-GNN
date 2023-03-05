import subprocess
import re
import sys
import os, pwd
import git
import socket

def get_git_info():
    repo = git.Repo(search_parent_directories = True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()
    return sha,dirty

def check_path():
    path_set = False
    for p in sys.path:
        if ROOT_DIR in p:
            path_set = True
    if (not path_set):
        sys.path.append(ROOT_DIR)

import os, pwd

from utils.utils import *
from normalize import *

def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

def parse_float(string, output):
    matches = re.findall(string,output)
    assert(len(matches) == 1)
    return "{:.2f}".format(float(matches[0]))


# measures cost of memory transfer of dataset
def run_naive(graphname, model, epochs, hidden_size, fsize, minibatch_size):
    output = subprocess.run(["python3",\
            "{}/quiver/gpu_sample_naive_dgl_gcn.py".format(ROOT_DIR),\
            "--graph",graphname,  \
            "--model", model , \
            "--num-hidden",  str(hidden_size), \
            "--batch-size", str(minibatch_size), \
                 "--num-epochs", str(epochs)] \
                , capture_output = True)

    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
        accuracy  = parse_float("accuracy:(\d+\.\d+)",out)
        epoch = parse_float("epoch_time:(\d+\.\d+)",out)
        sample_get  = parse_float("sample_time:(\d+\.\d+)",out)
        #movement_data_time = parse_float("data movement time:(\d+\.\d+)",out)
        movement_graph =  parse_float("movement graph:(\d+\.\d+)",out)
        movement_feat = parse_float("movement feature:(\d+\.\d+)",out)
        movement_data_time = movement_feat
        forward_time = parse_float("forward time:(\d+\.\d+)",out)
        backward_time = parse_float("backward time:(\d+\.\d+)",out)
        edges = parse_float("edges_per_epoch:(\d+\.\d+)",out)
        data_moved = parse_float("data moved:(\d+\.\d+)MB",out)
        edges = int(float(edges))
        data_moved = int(float(data_moved))
        #print("accuracy",accuracy)
        #print("edges", edges)
        #sample_get, movement_data_time, forward_time, backward_time = normalize(epoch, sample_get, movement_data_time, forward_time, backward_time)
    except Exception as e:
        with open('exception_naive.txt','a') as fp:
            fp.write(error)
        sample_get = "error"
        movement_graph = "error"
        movement_feat = "error"
        forward_time = "error"
        backward_time = "error"
        accuracy = "error"
        epoch = "error"
        movement_data_time = "error"
        data_moved = "error"
        edges = "error"
    return {"forward":forward_time, "sample_get":sample_get, "backward":backward_time, \
            "movement_graph":movement_graph, "movement_feat": movement_feat, "epoch":epoch,
            "accuracy": accuracy, "movement_data_time":movement_data_time, "data_moved":data_moved, "edges":edges}
