
def hello():
    print("World !!!!!!!!!")

import subprocess
import re
import sys
import os
import git
# Check environment before getting root dir.
import os, pwd
from utils.utils import *


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
        print(sys.path)
        sys.path.append(ROOT_DIR)
        print("Setting Path")
        print(sys.path)
'''

def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def check_single(ls):
    assert(len(ls) == 1)
    return ls[0]

def run_occ(graphname, model, cache_per, hidden_size, fsize, minibatch_size, \
        num_layers, num_partition, fanout):
    output = subprocess.run(["python3",\
            "{}/cu_train/main.py".format(ROOT_DIR),\
        "--graph",graphname,  \
        "--model", model , \
        "--cache-per" , str(cache_per),\
        "--num-hidden",  str(hidden_size), \
        "--batch-size", str(minibatch_size) ,\
        "--num-epochs", "6",\
        "--num-layers", str(num_layers), \
        "--num-gpus", str(num_partition),\
        "--fan-out", fanout, "--optimization1"\
        ] \
            , capture_output = True)
    # print(out,error)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
    #if True:
        accuracy  = check_single(re.findall("accuracy:(\d+\.\d+)",out))
        epoch = check_single(re.findall("epoch_time:(\d+\.\d+)",out))
        sample_get  = check_single(re.findall("sample_time:(\d+\.\d+)",out))
        movement_graph =  check_single(re.findall("movement graph:(\d+\.\d+)",out))
        movement_feat = check_single(re.findall("movement feature:(\d+\.\d+)",out))
        forward_time = check_single(re.findall("forward time:(\d+\.\d+)",out))
        backward_time = check_single(re.findall("backward time:(\d+\.\d+)",out))
        data_moved = check_single(re.findall("data movement:(\d+\.\d+)MB",out))
        edges_moved = re.findall("edges per epoch:(\d+\.\d+)",out)
        s = 0
        if(num_partition == -1):
            num_partition = 4
        for i in range(num_partition):
            s = s + float(edges_moved[i])
        edges_moved = s / num_partition
        sample_get = "{:.2f}".format(float(sample_get))
        movement_graph = "{:.2f}".format(float(movement_graph))
        movement_feat = "{:.2f}".format(float(movement_feat))
        accuracy = "{:.2f}".format(float(accuracy))
        forward_time = "{:.2f}".format(float(forward_time))
        epoch = "{:.2f}".format(float(epoch))
        backward_time = "{:.2f}".format(float(backward_time))
        data_moved = int(float(data_moved))
        edges_moved = int(float(edges_moved))

    except Exception as e:
        with open('exception_occ.txt','w') as fp:
            fp.write(error)

        sample_get = "error"
        movement_graph = "error"
        movement_feat = "error"
        forward_time = "error"
        backward_time = "error"
        accuracy = "error"
        epoch = "error"
        data_moved = "error"
        edges_moved = "error"
    return {"forward":forward_time, "sample_get":sample_get, "backward":backward_time, \
            "movement_graph":movement_graph, "movement_feat": movement_feat, "epoch":epoch,
                "accuracy": accuracy, "data_moved":data_moved, "edges_moved":edges_moved}
