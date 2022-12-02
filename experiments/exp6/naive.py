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
def run_quiver(graphname, model, epochs, hidden_size, fsize, minibatch_size):
    output = subprocess.run(["python3",\
            "{}/quiver/naive_dgl_gcn.py".format(ROOT_DIR),\
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
        movement_data_time = parse_float("data movement time:(\d+\.\d+)",out)
        movement_graph =  parse_float("movement graph:(\d+\.\d+)",out)
        movement_feat = parse_float("movement feature:(\d+\.\d+)",out)
        forward_time = parse_float("forward time:(\d+\.\d+)",out)
        backward_time = parse_float("backward time:(\d+\.\d+)",out)
        edges = parse_float("edges_per_epoch:(\d+\.\d+)",out)
        data_moved = parse_float("data moved:(\d+\.\d+)MB",out)
        edges = int(float(edges))
        data_moved = int(float(data_moved))
        #print("accuracy",accuracy)
        #print("edges", edges)
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
    return {"forward":forward_time, "sample_get":sample_get, "backward":backward_time, \
            "movement_graph":movement_graph, "movement_feat": movement_feat, "epoch":epoch,
            "accuracy": accuracy, "movement_data_time":movement_data_time, "data_moved":data_moved, "edges":edges}


def run_experiment_quiver( model ):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                ("ogbn-arxiv",16, 128, 1024), \
                # ("ogbn-arxiv",16, 128, 4096), \
                # ("ogbn-arxiv",16, 128, 256),  \
                # ("ogbn-products",16, 100, 1024), \
                # ("ogbn-products",16, 100, 1024), \
                # ("ogbn-products",16, 100, 4096), \
                # ("ogbn-products",16, 100, 256),  \
                # ("reorder-papers100M", 16, 128,  256),\
                # ("reorder-papers100M", 16, 128, 4096),\
                #("reorder-papers100M", 16, 128, 1024),\
                # ("amazon", 16, 200, 256),\
                # ("amazon", 16, 200,4096),\
                #("amazon", 16, 200, 1024),\
                 ]
    no_epochs = 5
    # settings = [("ogbn-arxiv",16, 128, 1024)]
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    #cache_rates = [".25"]
    #settings = [settings[0]]
    check_path()
    print(settings)
    sha,dirty = get_git_info()

    with open('{}/exp6_{}_naive.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  |" + \
            " batch-size | model  | sample_get | move-data | forward |" +\
              " backward  | epoch_time | accuracy | data_movement | edges \n")
    for graphname, hidden_size, fsize, batch_size in settings:
        out = run_quiver(graphname, model ,no_epochs, hidden_size, fsize, batch_size)
        print(out)
        with open('{}/exp6_{}_naive.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
            fp.write(("{} | {} | {} | {} | {} "+\
                   "| {} | {} | {} | {} | {} |"+\
                   " {} | {}  | {} | {} \n").format(graphname , SYSTEM , hidden_size, fsize,\
                    4 * batch_size, model, out["sample_get"], out["movement_data_time"] \
                    , out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                    out["data_moved"], out["edges"]))




if __name__=="__main__":
    run_experiment_quiver("GAT")
    run_experiment_quiver("GCN")
