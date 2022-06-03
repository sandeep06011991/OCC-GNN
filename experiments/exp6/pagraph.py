import subprocess
import re

def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def run_pagraph(graphname, epochs,cache_per, hidden_size, minibatch_size, fsize ):
    output = subprocess.run(["python3",\
            "/home/spolisetty/OCC-GNN/python/pa_cache_multi_gpu.py",
        "--graph",graphname, "--cache-per", cache_per, \
         "--num-epochs", str(epochs) , "--num-hidden" , str(hidden_size)
            ,"--fsize", str(fsize), "--batch-size", str(minibatch_size)], capture_output = True)
    out = str(output.stdout)
    error = str(output.stderr)
    #print(out,error)
    #if True:
    try:
        cache_hit  = re.findall("avg cache hit rate: (\d+\.\d+)",out)
        print(cache_hit)
        cache_hit = average_string(cache_hit)
        cache_hit = "{:0.2f}".format(float(cache_hit))
        forward =  re.findall("Avg forward backward time: (\d+\.\d+)sec",out)
        forward = average_string(forward)
        forward = "{:0.2f}".format(float(forward))
        movement = average_string(re.findall("avg move time: (\d+\.\d+)sec",out))
        movement = "{:0.2f}".format(float(movement))
        epoch_time = average_string(re.findall("avg epoch time: (\d+\.\d+)sec",out))
        epoch_time = "{:0.2f}".format(float(epoch_time))
    except:
        cache_hit = "error"
        forward = "error"
        movement = "error"
        epoch_time = "error"
    return {"forward":forward, "cache_hit":cache_hit, \
            "movement":movement, "epoch_time": epoch_time}


def run_experiment_pagraph():
    # Graph, num_epochs, hidden, fsize,  batch_size 
    settings = [\
               ("ogbn-arxiv",3, 32, -1, 4096),\
               ("ogbn-arxiv",3, 256, -1, 4096),\
               ("ogbn-arxiv",3, 32 , -1 , 1024),\
               ("ogbn-products",3, 32, -1, 4096),\
               ("ogbn-products",3, 256, -1, 4096),\
               ("ogbn-products",3, 32 , -1 , 1024),\
               ("com-youtube", 3, 32, 256, 4096),\
               ("com-youtube",3,32 ,1024, 4096)\
                #("ogbn-products",2), \
                # ("ogbn-papers100M",2), \
                # ("com-friendster",2), \
                # ("com-orkut",2) \
                ]
    # settings = [("ogbn-papers100M",2)]
    settings = [settings[0]]
    print("run settings", settings)
    cache_rates = [".05",".10",".24",".5"]
    cache_rates = [".05",".24", ".5"]
    cache_rates = [".25"]

    with open('exp6_pagraph.txt','a') as fp:
        fp.write("graph | hidden-size | batch_size | fsize  \
        | cached-gper-gpu | forward + backward(s)| pa-move(s) | epoch-time(s) | avg-cache hit rate \n")
    for graphname,no_epochs, hidden_size, fsize, batch_size in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .25:
                    continue
            out = run_pagraph(graphname, no_epochs, cache, hidden_size, batch_size, fsize)
            with open('exp6_pagraph.txt','a') as fp:
                fp.write("{} | {} | {} | {} | {} | {} | {} | {} | {}\n".format(graphname , \
                    hidden_size, batch_size, fsize, cache, \
                    out["forward"], out["movement"],  out["epoch_time"], out["cache_hit"]))




if __name__=="__main__":
    run_experiment_pagraph()
