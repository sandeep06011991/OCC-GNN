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

def run_occ(graphname, model, cache_per, hidden_size, fsize, minibatch_size, test_graph_dir):

    output = subprocess.run(["python3.8",\
            "{}/python/main.py".format(ROOT_DIR),\
        "--graph",graphname,  \
        "--model", model , \
        "--cache-per" , str(cache_per),\
        "--num-hidden",  str(hidden_size), \
        "--batch-size", str(minibatch_size) , "--num-epochs", "6", "--num-workers", "16", "--test-graph-dir", test_graph_dir] \
            , capture_output = True)
    # print(out,error)
    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
    #if True:
        test_accuracy = re.findall("test_accuracy_log:(\d+\.\d+), epoch:(\d+)",out)
        test_accuracy.sort(key = lambda x:int(x[1]))
        ret = []
        for i in test_accuracy:
            ret.append(i[0])

    except Exception as e:
        with open('exception_occ.txt','w') as fp:
            fp.write(error)
        ret = "error"
    return {"test_accuracy":ret }


def run_experiment_occ(model):
    # graph, num_epochs, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv",16, 128, 1024),
                # ("ogbn-arxiv", 16, 128, 4096), \
                #("ogbn-arxiv",16, 128, 16384),\
                ("ogbn-arxiv",3, 32 , -1 , 4096, "ogbn-arxiv"), \
                ("ogbn-products",16, 100, 4096,"ogbn-products"), \
                #("ogbn-products", 16, 100, 4096), \
                #("ogbn-products",16, 100 , 16384), \
                #("reorder-papers100M", 16, 128, 1024),\
                ("reorder-papers100M", 16, 128, 4096, "test_reorder_papers100M"),\
                ("amazon", 16, 200, 4096, "amazon"),\
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
    #cache_rates = ["0", ".10", ".25", ".50", ".75", "1"]
    #cache_rates = ["0", ".10", ".25"]
    #settings = [("ogbn-arxiv", 16, 128, 1024),]
    #cache_rates = [".10",".25"]
    cache_rates = [".25"]
    #settings = [settings[0]]
    #check_path()
    print(settings)
    sha,dirty = get_git_info()
    assert(model in ["gcn","gat"])
    with open(OUT_DIR + '/exp8_occ_{}.txt'.format(SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  | batch-size |"+\
            " model  | test_accuracy \n")
    for graphname, hidden_size, fsize, batch_size, test_graph in settings:
        for cache in cache_rates:
            if graphname in ["ogbn-papers100M","com-friendster"]:
                if float(cache) > .3:
                    continue
            out = run_occ(graphname, model,  cache, hidden_size,fsize, batch_size, test_graph)
            with open(OUT_DIR + '/exp8_occ_{}.txt'.format(SYSTEM),'a') as fp:
                fp.write("{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |{} | {} \n".\
                    format(graphname , SYSTEM, cache, hidden_size, fsize, batch_size, model, out["test_accuracy"])

'''
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
'''

if __name__ == "__main__":
    run_experiment_occ("gcn")
    run_experiment_occ("gat")
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
