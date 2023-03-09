from baselines.occ import run_occ
#from baselines.pagraph import *
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


def run_experiment_occ(model):
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv", 16, 128, 4096), \
                ("ogbn-products", 16, 100, 4096), \
                #("reorder-papers100M", 16, 128, 4096),\
                #("amazon", 16, 200, 4096),\
                #("reorder-papers100M", 16, 128, 4096),\
                #("amazon", 16, 200, 4096),\
                 ]
    cache  = ".25"
    num_layers = 3
    num_partition= 4
    #hidden_sizes = [16]
    fanouts = ["10,10,10", "20,20,20", "30,30,30"]
    #fanouts = ["10,10,10"]
    #hidden_sizes = [64]
    hidden_size = 16

    sha,dirty = get_git_info()
    assert(model in ["gcn","gat","gat-pull"])
    with open(OUT_DIR + '/fanouts/occ_{}.txt'.format(SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  | batch-size |"+\
                "num_partitions | num-layers |" + \
            " model  | fanout |  sample_get | move-graph | move-feature | forward | backward  |"+\
                " epoch_time | accuracy | data_moved | edges_computed\n")
    for graphname, hidden_size, fsize, batch_size in settings:
        for fanout in fanouts:
            out = run_occ(graphname, model,  cache, hidden_size, fsize,\
                    batch_size, num_layers, num_partition, fanout)
            with open(OUT_DIR + '/fanouts/occ_{}.txt'.format(SYSTEM),'a') as fp:
                fp.write("{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |{} | {} \n".\
                format(graphname , SYSTEM, cache, hidden_size, fsize, batch_size,\
                    num_partition, num_layers, model, fanout, out["sample_get"], \
                    out["movement_graph"], out["movement_feat"], out["forward"], out["backward"], \
                     out["epoch"], out["accuracy"], out["data_moved"], out["edges_moved"]))




if __name__ == "__main__":
    run_experiment_occ("gcn")
    run_experiment_occ("gat")
    # run_experiment_occ("gat-pull")
    print("Success!!!!!!!!!!!!!!!!!!!")
    #run_model("gat")
