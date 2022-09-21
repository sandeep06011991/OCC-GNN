import subprocess
import re
import sys
import os
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


def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c

# measures cost of memory transfer of dataset
def run_quiver(graphname, epochs,cache_per, hidden_size, fsize, minibatch_size):
    output = subprocess.run(["python3",\
            "/home/q91/OCC-GNN/quiver/gcn.py",\
        "--graph",graphname,  \
         "--model", model , \ 
         "--cache-per" , str(cache-per)])
                , capture_output = True)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    if True:
        accuracy  = re.findall("accuracy:(\d+\.\d+)",out)[0]
        epoch = re.findall("epoch:(\d+\.\d+)",out)[0]
        sample_get  = re.findall("sample_time:(\d+\.\d+)",out)[0]
        movement_graph =  re.findall("movement graph:(\d+\.\d+)",out)[0]
        movement_feat = re.findall("movement feature:(\d+\.\d+)",out)[0]
        forward_time = re.findall("forward time:(\d+\.\d+)",out)[0]
        backward_time = re.findall("backward time:(\d+\.\d+)",out)[0]
        
        sample_get = "{:.2f}".format(float(sample_get))
        movement_graph = "{:.2f}".format(float(movement_graph))
        movement_feat = "{:.2f}".format(float(movement_feat))
        accuracy = "{:.2f}".format(float(accuracy))
        forward_time = "{:.2f}".format(float(forward_time))
        epoch = "{:.2f}".format(float(epoch))
        backward_time = "{:.2f}".format(float(backward_time))
        move_time = "{:.2f}".format(float(move_time))

    #except:
        sample_get = "error"
        movement_graph = "error"
        movement_feat = "error"
        forward_time = "error"
        backward_time = "error"
        accuracy = "error"
        epoch = "error"
    return {"forward":forward_time, "sample_get":sample_get, "backward":backward_time, \
            "movement_graph":movement_graph, "movement_feat": movement_feat, "epoch":epoch, 
                "accuracy": accuracy}


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
    cache_rates = ["0", ".10", ".25", ".50", ".75", "1"]
    #settings = [settings[0]]
    check_path()
    print(settings)
    sha,dirty = get_git_info()
    with open('exp6_occ.txt','a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | hidden-size | fsize  | batch-size | model  | \
                sample_get | move | forward | backward  | epoch_time | accuracy \n")
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
