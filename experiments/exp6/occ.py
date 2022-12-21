import subprocess
import re
import sys
import os
import git
# Check environment before getting root dir.
import os, pwd
from utils.utils import *
'''
uname = pwd.getpwuid(os.getuid())[0]
os.environ['NCCL_BUFFSIZE'] = str(1024 * 1024 * 80)
if uname == 'spolisetty':
    ROOT_DIR = "/home/spolisetty/OCC-GNN/cslicer/"
    SRC_DIR = "/home/spolisetty/OCC-GNN/python/main.py"
    SYSTEM = 'jupiter'
    OUT_DIR = '/home/spolisetty/OCC-GNN/experiments/exp6/'
if uname == 'q91':
    ROOT_DIR = "/home/q91/OCC-GNN/cslicer/"
    SRC_DIR = "/home/q91/OCC-GNN/python/main.py"
    SYSTEM = 'ornl'
    OUT_DIR = '/home/q91/OCC-GNN/experiments/exp6/'
if uname == 'spolisetty_umass_edu':
    ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/cslicer"
    SRC_DIR = "/home/spolisetty_umass_edu/OCC-GNN/python/main.py"
    SYSTEM = 'unity'
    OUT_DIR = '/home/spolisetty_umass_edu/OCC-GNN/experiments/exp6/'
'''
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

def run_occ(graphname, model, cache_per, hidden_size, fsize, minibatch_size):

    output = subprocess.run(["python3.8",\
            "{}/python/main.py".format(ROOT_DIR),\
        "--graph",graphname,  \
        "--model", model , \
        "--cache-per" , str(cache_per),\
        "--num-hidden",  str(hidden_size), \
        "--batch-size", str(minibatch_size) , "--num-epochs", "6", "--num-workers", "16"] \
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
        for i in range(4):
            s = s + float(edges_moved[i])
        edges_moved = s / 4
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


def run_experiment_occ(model):
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv",16, 128, 1024),
                # ("ogbn-arxiv", 16, 128, 4096), \
                #("ogbn-arxiv",16, 128, 16384),\
                #("ogbn-arxiv",3, 32 , -1 , 1024), \
                #("ogbn-products",16, 100, 1024), \
                #("ogbn-products", 16, 100, 4096), \
                #("ogbn-products",16, 100 , 16384), \
                #("reorder-papers100M", 16, 128, 1024),\
                #("reorder-papers100M", 16, 128, 4096),\
                #("reorder-papers100M", 16, 128, 16384),\
                ("amazon", 16, 200, 1024),\
                #("amazon", 16, 200, 4096),\
                ("amazon", 16, 200, 16384),\
                #("com-youtube", 3, 32, 256, 4096),\
                #("com-youtube",3,32,1024, 4096)\
                # ("com-youtube",2), \
                # ("ogbn-products",2), \
                # ("ogbn-papers100M",2), \
                # ("com-friendster",2), \
                 # ("com-orkut",5, 256, 256, 4096) \
                 ]
    # settings = [("ogbn-products", 16, 128, 4096),]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    #cache_rates = ["0", ".10", ".25", ".50", ".75", "1"]
    #cache_rates = ["0",".1",".25"]
    #settings = [("ogbn-products", 16, 128, 4096),]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    cache_rates = ["0",  ".25", ".50", ".75", "1"]
    #cache_rates = ["0", ".10", ".25"]
    #settings = [("ogbn-arxiv", 16, 128, 1024),]
    cache_rates = [".25"]
    cache_rates = [".25"]
    #cache_rates = ["0", ".10", ".25", ".50", ".75", "1"]
    #cache_rates = ["0", ".10", ".25"]
    #settings = [("ogbn-arxiv", 16, 128, 1024),]
    #cache_rates = [".10",".25"]
    #cache_rates = [".25"]
    #settings = [settings[0]]
    #check_path()
    print(settings)
    sha,dirty = get_git_info()
    assert(model in ["gcn","gat","gat-pull"])
    with open(OUT_DIR + '/exp6/exp6_occ_{}.txt'.format(SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  | batch-size |"+\
            " model  | sample_get | move-graph | move-feature | forward | backward  |"+\
                " epoch_time | accuracy | data_moved | edges_computed\n")
    for graphname, hidden_size, fsize, batch_size in settings:
        for cache in cache_rates:
            if batch_size != 4096:
                if cache != ".10":
                    pass
                    #continue
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .3:
                    continue
            out = run_occ(graphname, model,  cache, hidden_size,fsize, batch_size)
            with open(OUT_DIR + '/exp6/exp6_occ_{}.txt'.format(SYSTEM),'a') as fp:
                fp.write("{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |{} | {} \n".\
                    format(graphname , SYSTEM, cache, hidden_size, fsize, batch_size, model, out["sample_get"], \
                        out["movement_graph"], out["movement_feat"], out["forward"], out["backward"], \
                         out["epoch"], out["accuracy"], out["data_moved"], out["edges_moved"]))


def run_sbatch(model, settings, cache_rates):
    check_path()
    print(settings)
    sha,dirty = get_git_info()
    assert(model in ["gcn","gat"])
    with open(OUT_DIR + 'exp6_occ_{}.txt'.format(SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  | batch-size |"+\
            " model  | sample_get | move-graph | move-feature | forward | backward  |"+\
                " epoch_time | accuracy | data_moved | edges_computed\n")
    for graphname, hidden_size, fsize, batch_size in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .3:
                    continue
            out = run_occ(graphname, model,  cache, hidden_size,fsize, batch_size)
            with open(OUT_DIR + 'exp6_occ_{}.txt'.format(SYSTEM),'a') as fp:
                fp.write("{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |{} | {} \n".\
                    format(graphname , SYSTEM, cache, hidden_size, fsize, batch_size, model, out["sample_get"], \
                        out["movement_graph"], out["movement_feat"], out["forward"], out["backward"], \
                         out["epoch"], out["accuracy"], out["data_moved"], out["edges_moved"]))


if __name__ == "__main__":
    run_experiment_occ("gcn")
    run_experiment_occ("gat")
    #run_experiment_occ("gat-pull")
    print("Success!!!!!!!!!!!!!!!!!!!")
    #run_model("gat")
    assert(False)
    #return
    import argparse
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv", required = True)
    # training details
    # model name and details
    argparser.add_argument('--cache-per', type = str)
    argparser.add_argument('--model',help="gcn|gat")
    argparser.add_argument('--batch-size', type=int)
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    if args.graph == "all":
        run_experiment_occ("gcn")
        run_expoeriment_occ("gat")
    else:
        dataset = args.graph
        cache_per = args.cache_per
        batch_size = args.batch_size
        model = args.model
        assert(dataset != None) and (cache_per != None) and (model != None)
        fdict = {"ogbn-arxiv":128, "ogbn-products":100, "reorder-papers100M":128, "amazon":200}
        settings = [(dataset,"16", fdict[dataset], batch_size)]
    #print(cache_per)
        run_sbatch(model, settings, [cache_per])
