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
def run_quiver(graphname, model, epochs, hidden_size, fsize, minibatch_size, test_graph_dir):
    output = subprocess.run(["python3",\
            "{}/quiver/naive_dgl_gcn.py".format(ROOT_DIR),\
            "--graph",graphname,  \
            "--model", model , \
            "--num-hidden",  str(hidden_size), \
            "--batch-size", str(minibatch_size), \
                 "--num-epochs", str(epochs), \
                "--test-graph", str(test_graph_dir) ] \
                , capture_output = True)

    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
        test_accuracy = re.findall("Test Accuracy:(\d+\.\d+) Epoch:(\d+)",out)
        test_accuracy.sort(key = lambda x:int(x[1]))
        ret = []
        for i in test_accuracy:
            ret.append(i[0])
            
    except Exception as e:
        with open('exception_naive.txt','a') as fp:
            fp.write(error)
        ret = "error"
    return {"test_accuracy": ret}


def run_experiment_naive(model):
    # graph, hidden_size, fsize, minibatch_size, ogbn-arxiv
    settings = [
                ("ogbn-arxiv", 16, 128, 1024, "ogbn-arxiv"), \
                ("ogbn-products",16, 100, 1024, "ogbn-products"), \
                ("reorder-papers100M", 16, 128, 1024, "test_reorder_papers100M"),\
                # ("amazon", 16, 200, 1024, "amazon"),\
                 ]
    no_epochs = 50
    # settings = [("ogbn-arxiv",16, 128, 1024)]
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    #cache_rates = [".25"]
    #settings = [settings[0]]
    check_path()
    print(settings)
    sha,dirty = get_git_info()

    with open('{}/exp8/exp8_{}_naive.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  |" + \
            " batch-size | model  | sample_get | move-data | forward |" +\
              " backward  | epoch_time | accuracy | data_movement | edges \n")
    for graphname, hidden_size, fsize, batch_size, test_graph in settings:
        out = run_quiver(graphname, model ,no_epochs, hidden_size, fsize, batch_size, test_graph)
        print(out)
        with open('{}/exp8/exp8_{}_naive.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
            fp.write(("{} | {} | {} | {} | {} "+\
                   "| {} | {} \n").format(graphname , SYSTEM , hidden_size, fsize,\
                    4 * batch_size, model, out["test_accuracy"]))




if __name__=="__main__":
    run_experiment_naive("GCN")
    run_experiment_naive("GAT")
