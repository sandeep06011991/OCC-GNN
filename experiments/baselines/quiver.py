import subprocess
import re
import sys
import os
import git
import socket

#ROOT_DIR ="/home/q91/torch-quiver/srcs/python"

def get_git_info():
    repo = git.Repo(search_parent_directories = True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()
    return sha,dirty
'''
def check_path():
    path_set = False
    for p in sys.path:
        if ROOT_DIR in p:
            path_set = True
    if (not path_set):
        sys.path.append(ROOT_DIR)
'''
from utils.utils import *

def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

def parse_float(string, output):
    matches = re.findall(string,output)
    print(matches)
    assert(len(matches) == 1)
    return "{:.2f}".format(float(matches[0]))


# measures cost of memory transfer of dataset
def run_quiver(graphname, model, epochs,cache_per, hidden_size,\
        fsize, minibatch_size, num_layers, fanout):
    output = subprocess.run(["python3",\
            "{}/quiver/dgl_gcn.py".format(ROOT_DIR),\
            "--graph",graphname,  \
            "--model", model , \
            "--cache-per" , str(cache_per),\
            "--num-hidden",  str(hidden_size), \
            "--batch-size", str(minibatch_size), \
            "--num-hidden", str(num_layers),\
            "--fan-out", fanout, \
            "--data", "quiver", "--num-epochs", str(epochs), "--sample-gpu"] \
                , capture_output = True)

    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
    #if True:
        accuracy  = parse_float("accuracy:(\d+\.\d+)",out)
        epoch = parse_float("epoch_time:(\d+\.\d+)",out)
        sample_get  = parse_float("sample_time:(\d+\.\d+)",out)
        movement_data = parse_float("movement data time:(\d+\.\d+)", out)
        movement_graph =  parse_float("movement graph:(\d+\.\d+)",out)
        movement_feat = parse_float("movement feature:(\d+\.\d+)",out)
        forward_time = parse_float("forward time:(\d+\.\d+)",out)
        backward_time = parse_float("backward time:(\d+\.\d+)",out)[0]
        edges = re.findall("edges_per_epoch:(\d+\.\d+)",out)
        edges = [float(edge) for edge in edges]
        assert(len(edges) == 4)
        edges = sum(edges)/4

        data_moved = re.findall("data moved :(\d+\.\d+)MB",out)
        data_moved = [float(d) for d in data_moved]
        assert(len(data_moved) == 4)
        data_moved = sum(data_moved)/4
        #print("accuracy",accuracy)
        #print("edges", edges)
        #sample_get, movement_data, forward_time, backward_time = normalize\
        #        (epoch, sample_get, movement_data, forward_time, backward_time)

    except Exception as e:
        with open('exception_quiver.txt','a') as fp:
            fp.write(error)
        sample_get = "error"
        movement_graph = "error"
        movement_feat = "error"
        movement_data = "error"
        forward_time = "error"
        backward_time = "error"
        accuracy = "error"
        epoch = "error"
        data_moved = "error"
    return {"forward":forward_time, "sample_get":sample_get, "backward":backward_time, \
            "movement_graph":movement_graph, "movement_data": movement_data, \
                "movement_feat": movement_feat, "epoch":epoch,
                "accuracy": accuracy,"data_moved":data_moved, "edges":edges}



if __name__=="__main__":
    run_quiver("ogbn-arxiv", "GCN", 4 , .20, 16, 128, 1024, 3, "20,20,20") 
    #run_experiment_quiver("GAT")
    #run_experiment_quiver("GCN")
