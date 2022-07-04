# import dgl
# import torch
import time
import subprocess


import re
def run_naive_experiment(filename):
    output = subprocess.run(["python3","../../python/no_cache_multi_gpu.py","--graph",
            filename], capture_output=True)
    output = str(output.stdout)
    print(output)
    movement = float(re.findall(r"data movement_time:(\d+\.\d+)",output)[0])
    compute1 = float(re.findall(r"forward_time:(\d+\.\d+)",output)[0])
    return {"movement":movement,"compute1":compute1}

def run_metis_experiment(filename):
    output = subprocess.run(["python3","../../python/batch_slice_multi_gpu.py","--graph",
            filename], capture_output=True)
    output = str(output.stdout)
    print(output)
    forward  = float(re.findall("forward_time_per_epoch:(\d+\.\d+)",output)[0])
    merge =  float(re.findall("merge_time per epoch:(\d+\.\d+)",output)[0])
    movement = float(re.findall("data transfer:(\d+\.\d+)",output)[0])
    slice = float(re.findall("graph splitting time:(\d+\.\d+)",output)[0])

    return {"forward":forward, "merge":merge, "movement":movement,"slice":slice}

def run_experiment():
    partition_scheme = ["random","metis","optimum"]
    partition_scheme = ["metis"]
    filename = ["pubmed","reddit","ogbn-arxiv","ogbn-products"]
    filename = ["ogbn-products","reddit"]
    # filename = ["ogbn-arxiv"]
    with open("exp5.txt",'a') as fp:
        fp.write("GRAPH | PARTITION | MOVE | AGGR | MERGE | SLICE\n")
    for f in filename:
        out = run_naive_experiment(f)
        with open("exp5.txt",'a') as fp:
            fp.write("{} | {} | {:.4f} | {:.4f} | 0 | 0\n".format(f, "naive",out["movement"],out["compute1"]))
        out = run_metis_experiment(f)
        with open("exp5.txt",'a') as fp:
            fp.write("{} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} \n".format(f, "metis",\
                out["movement"],out["forward"],out["merge"],out["slice"]))



# python3 run.py "reddit|cora|pubmed" "hops"
if __name__=="__main__":
    # import sys
    # assert(len(sys.argv) == 2)
    # graphName = sys.argv[1]
    # # hops = int(sys.argv[2])
    # # graphName = "cora"
    # # hops = 2
    run_experiment()
    # run_naive_experiment(graphName)
    # run_experiment(graphName,hops)
