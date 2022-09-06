import subprocess
import re
import sys
import os
import git

ROOT_DIR = "/home/spolisetty/OCC-GNN/cslicer/"

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


def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def run_occ(graphname, epochs,cache_per, hidden_size, fsize, minibatch_size):
    output = subprocess.run(["python3",\
            "/home/spolisetty/OCC-GNN/python/cslicer_sharer.py",\
        "--graph",graphname,  \
         "--num-epochs", str(epochs) , "--num-hidden" , str(hidden_size),\
            "--fsize", str(fsize) , "--batch-size", str(minibatch_size)], capture_output = True)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    if True:
    #try:
        sample_get  = re.findall("avg sample get time: (\d+\.\d+)sec",out)[0]
        forward =  re.findall("avg forward time: (\d+\.\d+)sec",out)[0]
        epoch_time = re.findall("avg epoch time: (\d+\.\d+)sec",out)[0]
        back_time = re.findall("avg backward time: (\d+\.\d+)sec",out)[0]
        move_time = re.findall("avg move time: (\d+\.\d+)sec",out)[0]
        #movement = re.findall("cache refresh time (\d+\.\d+)",out)[0]
        sample_get = "{:.2f}".format(float(sample_get))
        forward = "{:.2f}".format(float(forward))
        epoch_time = "{:.2f}".format(float(epoch_time))
        back_time = "{:.2f}".format(float(back_time))
        move_time = "{:.2f}".format(float(move_time))

    #except:
        slice_get = "error"
        forward = "error"
        move_time = "error"
        epoch_time = "error"
        back_time = "error"
    return {"forward":forward, "sample_get":sample_get, "back_time":back_time, \
            "move_time":move_time, "epoch_time":epoch_time}


def run_experiment_occ(settings = None):
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    settings = [
                 ("ogbn-arxiv",3, 32, -1, 4096), \
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
                 ]
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    cache_rates = [".25"]
    #settings = [settings[0]]
    check_path()
    print(settings)
    sha,dirty = get_git_info()
    with open('exp6_occ.txt','a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | hidden-size | fsize  | batch-size |\
                sample_get | move | forward | backward  | epoch_time \n")
    for graphname,no_epochs,hidden_size, fsize, batch_size in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .3:
                    continue
            out = run_occ(graphname, no_epochs, cache, hidden_size,fsize, batch_size)
            with open('exp6_occ.txt','a') as fp:
                fp.write("{} | {} | {} | {} | {} | \
                          {} | {} | {} | {}  \n".format(graphname , \
                                hidden_size, fsize, batch_size,\
                                 out["sample_get"], out["move_time"],  \
                    out["forward"], out["back_time"],  out["epoch_time"]))




if __name__=="__main__":
    run_experiment_occ()
