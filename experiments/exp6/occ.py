import subprocess
import re

def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def run_occ(graphname, epochs,cache_per, hidden_size):
    output = subprocess.run(["python3",\
            "/home/spolisetty/OCC-GNN/python/train.py",
        "--graph",graphname, "--cache-per", cache_per, \
         "--num-epochs", str(epochs) , "--num-hidden" , str(hidden_size)
            ], capture_output = True)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    try:
        slice_time  = re.findall("batch slice time (\d+\.\d+)",out)[0]
        forward =  re.findall("avg forward time (\d+\.\d+)",out)[0]
        movement = re.findall("cache refresh time (\d+\.\d+)",out)[0]
    except:
        slice_time = "error"
        forward = "error"
        movement = "error"
    return {"forward":forward, "slice_time":slice_time, \
        "movement":movement}


def run_experiment_occ():
    settings = [\
                ("ogbn-arxiv",10), \
                # ("com-youtube",2), \
                ("ogbn-products",2), \
                # ("ogbn-papers100M",2), \
                # ("com-friendster",2), \
                # ("com-orkut",5) \
                ]
    # settings = [("ogbn-papers100M",2)]
    hidden_size = 128
    # cache_rates = [".05",".10",".24",".5"]
    cache_rates = [".05",".24", ".5"]
    cache_rates = [".24"]

    with open('exp6.txt','a') as fp:
        fp.write("graph | hidden-size | cached-gper-gpu | occ-forward | occ-move | occ-slice-time \n")
    for graphname,no_epochs in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .3:
                    continue
            out = run_occ(graphname, no_epochs, cache, hidden_size)
            with open('exp6.txt','a') as fp:
                fp.write("{} | {} | {} | {} | {} | {}\n".format(graphname , \
                                hidden_size, cache, \
                    out["forward"], out["movement"],  out["slice_time"]))




if __name__=="__main__":
    run_experiment()
