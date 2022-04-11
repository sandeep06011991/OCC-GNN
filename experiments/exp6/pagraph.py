import subprocess
import re

def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def run_pagraph(graphname, epochs,cache_per, hidden_size):
    output = subprocess.run(["python3",\
            "/home/spolisetty/OCC-GNN/python/pa_cache_multi_gpu.py",
        "--graph",graphname, "--cache-per", cache_per, \
         "--num-epochs", str(epochs) , "--num-hidden" , str(hidden_size)
            ], capture_output = True)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    try:
        cache_hit  = re.findall("avg cache hit rate: (\d+\.\d+)",out)
        cache_hit = average_string(cache_hit)
        forward =  re.findall("Avg forward time: (\d+\.\d+)sec",out)
        forward = average_string(forward)
        movement = average_string(re.findall("avg move time: (\d+\.\d+)sec",out))
    except:
        cache_hit = "error"
        forward = "error"
        movement = "error"
    return {"forward":forward, "cache_hit":cache_hit, \
        "movement":movement}


def run_experiment_pagraph():
    settings = [\
                # ("ogbn-arxiv",10), \
                # ("com-youtube",10), \
                ("ogbn-products",5), \
                # ("ogbn-papers100M",2), \
                # ("com-friendster",2), \
                ("com-orkut",5) \
                ]
    # settings = [("ogbn-papers100M",2)]
    hidden_size = 128
    cache_rates = [".05",".10",".24",".5"]
    cache_rates = [".05",".24", ".5"]
    cache_rates = [".20"]

    with open('exp6.txt','a') as fp:
        fp.write("graph | hidden-size | cached-gper-gpu | pa-forward | pa-move | avg-cache hit rate \n")
    for graphname,no_epochs in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .25:
                    continue
            out = run_pagraph(graphname, no_epochs, cache, hidden_size)
            with open('exp6.txt','a') as fp:
                fp.write("{} | {} | {} | {} | {} | {}\n".format(graphname , \
                                hidden_size, cache, \
                    out["forward"], out["movement"],  out["cache_hit"]))




if __name__=="__main__":
    run_experiment_pagraph()
