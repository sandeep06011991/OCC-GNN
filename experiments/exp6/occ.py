import subprocess
import re

import os

os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/cslicer/"


def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def run_occ(graphname, epochs,cache_per, hidden_size, fsize, minibatch_size):
    output = subprocess.run(["python3",\
            "/home/spolisetty/OCC-GNN/python/train.py",\
        "--graph",graphname, "--cache-per", cache_per, \
         "--num-epochs", str(epochs) , "--num-hidden" , str(hidden_size),\
            "--fsize", str(fsize) , "--batch-size", str(minibatch_size)], capture_output = True)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #if True:
    try:
        slice_time  = re.findall("batch slice time (\d+\.\d+)",out)[0]
        forward =  re.findall("avg forward time (\d+\.\d+)",out)[0]
        epoch_time = re.findall("avg epoch time (\d+\.\d+)",out)[0]
        #movement = re.findall("cache refresh time (\d+\.\d+)",out)[0]
        slice_time = "{:.2f}".format(float(slice_time))
        forward = "{:.2f}".format(float(forward))
        movement = 0
    except:
        slice_time = "error"
        forward = "error"
        movement = "error"
        epoch_time = "error"
    return {"forward":forward, "slice_time":slice_time, \
            "movement":movement, "epoch_time":epoch_time}


def run_experiment_occ(settings):
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    # settings = [
                # ("ogbn-arxiv",3, 32, -1, 4096), \
                #("ogbn-arxiv",3, 256, -1, 4096),\
                #("ogbn-arxiv",3, 32 , -1 , 1024), \
                #("ogbn-products",3, 32, -1, 4096), \
                # ("ogbn-products",3, 256, -1, 4096), \
                #("ogbn-products",3, 32 , -1 , 1024), \
                #("com-youtube", 3, 32, 256, 4096),\
                #("com-youtube",3,32,1024, 4096)\
                # ("com-youtube",2), \
                # ("ogbn-products",2), \
                # ("ogbn-papers100M",2), \
                # ("com-friendster",2), \
                 # ("com-orkut",5, 256, 256, 4096) \
                # ]
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    cache_rates = [".05",".24", ".5"]
    cache_rates = [".25"]
    #settings = [settings[0]]
    print(settings)
    with open('exp6_occ.txt','a') as fp:
        fp.write("graph | hidden-size | fsize  | batch-size |\
 cached-gper-gpu | fp_bp(s) | cache_refresh(s) | occ-slice-time(s) | epoch_time \n")
    for graphname,no_epochs,hidden_size, fsize, batch_size in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .3:
                    continue
            out = run_occ(graphname, no_epochs, cache, hidden_size,fsize, batch_size)
            with open('exp6_occ.txt','a') as fp:
                fp.write("{} | {} | {} | {} | {} |  {} | {} | {} | {} \n".format(graphname , \
                                hidden_size, fsize, batch_size, cache, \
                    out["forward"], out["movement"],  out["slice_time"], out["epoch_time"]))




if __name__=="__main__":
    run_experiment_occ()
