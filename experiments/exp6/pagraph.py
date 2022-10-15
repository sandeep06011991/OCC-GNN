 # Ensure cuda/10.2 is loaded
import subprocess
import os
import time

ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/pagraph"
DATA_DIR = "/home/spolisetty_umass_edu/OCC-GNN/experiments/exp6/"
import sys
import re
import git

Feat = {"ogbn-arxiv":"128","ogbn-products":"100", "reorder-papers100M":"128","amazon":200}

def get_git_info():
    repo = git.Repo(search_parent_directories = True)
    sha = repo.head.object.hexsha
    dirty = repo.is_dirty()
    return sha,dirty

def check_path():
    path_set = False
    for p in sys.path:
        print(p)
        if ROOT_DIR ==  p:
            path_set = True
    if (not path_set):
        print("Setting Path")
        sys.path.append(ROOT_DIR)
def avg(ls):
    return (sum(ls[1:]/(len(ls)-1)))

def check_no_stale():
    ps = os.popen('ps').read().split('\n')
    py_process = 0
    for p in ps:
        if 'python3' in p:
            py_process += 1
    if py_process != 1:
        print("stale processes exist clean up")
    assert(py_process == 1)
    # I can add a kill ad clean up later.


# Start server and wait for ready
def start_server(filename):
    cmd = ['python3','{}/server/pa_server.py'.format(ROOT_DIR),'--dataset',filename]
    print(cmd)
    #cmd = ['python3','hello.py']
    fp = subprocess.Popen(cmd,
                    stdout = subprocess.PIPE,
                     stderr = subprocess.PIPE,
                     text = True)
    os.set_blocking(fp.stdout.fileno(), False)
    os.set_blocking(fp.stderr.fileno(),False)
    while(True):
        out = fp.stdout.readline()
        err = fp.stderr.readline()
        print(out,err)
        sys.stdout.flush()
        if 'start running graph' in str(out):
            print("Breaking")
            break
        time.sleep(1)
    print("Server is running can start client Finally")
    sys.stdout.flush()
    return fp

def start_client(filename, model, num_hidden, batch_size, cache_per):
    feat_size = Feat[filename]
    print(cache_per)
    cmd = ['python3','{}/examples/profile/pa_gcn.py'.format(ROOT_DIR), '--dataset', filename,'--n-epochs','3'\
                    , '--n-hidden', str(num_hidden), '--batch-size', str(batch_size),\
                        '--model', model, "--cache-per",str(cache_per)]
    output = subprocess.run(cmd, capture_output=True)
    out = str(output.stdout)
    err = str(output.stderr)
    print(out,err)
    output = out
    accuracy = float(re.findall("accuracy: (\d+\.\d+)", output)[0])
    move_graph = float(re.findall("movement graph: (\d+\.\d+)s", output)[0])
    forward = float(re.findall("forward time: (\d+\.\d+)s", output)[0])
    backward = float(re.findall("backward time: (\d+\.\d+)s", output)[0])
    sample = float(re.findall("Sample time: (\d+\.\d+)s",output)[0])
    #compute  = float(re.findall("Compute time: (\d+\.\d+)s",output)[0])
    collect  = float(re.findall("CPU collect: (\d+\.\d+)s",output)[0])
    move  = float(re.findall("CUDA move: (\d+\.\d+)s",output)[0])
    epoch  = float(re.findall("Epoch time: (\d+\.\d+)s",output)[0])
    miss_rate = float(re.findall("Miss rate: (\d+\.\d+)s",output)[0])
    miss_num = int(float(re.findall("Miss num per epoch: (\d+\.\d+)MB, device \d+",output)[0]))
    edges_processed = int(float(re.findall("Edges processed per epoch: (\d+\.\d+)",output)[0]))
    return {"sample":sample, "forward": forward, "backward": backward,\
            "move_feat":"{:.3f}".format(move + collect), "epoch_time":epoch, "miss_rate": miss_rate, "move_graph":move_graph\
                , "accuracy": accuracy, "miss_num":miss_num, "edges":edges_processed}

def run_experiment_on_graph(filename, model, hidden_size, batch_size, cache_per):
    fp = start_server(filename)
    feat_size = Feat[filename]
    res = start_client(filename, model, hidden_size, batch_size, cache_per)
    WRITE = "{}/exp6_pagraph.txt".format(DATA_DIR)
    with open(WRITE,'a') as fp:
        fp.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(filename, "unity", cache_per, hidden_size, feat_size, \
                4 * batch_size, model, res["sample"], res["move_graph"], res["move_feat"],res["forward"], \
                    res["backward"],res["epoch_time"], res["accuracy"],res["miss_num"], res["edges"] ))
    fp.close()

def run_model(model):
    graphs = ['ogbn-arxiv','ogbn-products']
    #graphs = ['ogbn-papers100M']
    #graphs = ['ogbn-arxiv']
    settings = [('ogbn-arxiv', 16, 1024),
                #('ogbn-arxiv', 16, 256),
                #('ogbn-arxiv', 16, 4096),
                #('ogbn-products', 16, 1024),
                #('ogbn-products', 16, 256),
                #('ogbn-products', 16, 4096),
                #('reorder-papers100M',16, 1024), 
                #('reorder-papers100M',16, 256),
                #('reorder-papers100M',16, 4096)
                ]

    cache_per = ["0",".1",".25",".5","1"]
    cache_per = [ "0"]
    run_experiment(model, settings, cache_per)

    #settings = [('ogbn-arxiv',16,1024)]
def run_experiment(model, settings, cache_per):
    sha, dirty = get_git_info()
    check_path()
    check_no_stale()
    with open('{}/exp6_pagraph.txt'.format(DATA_DIR),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  | batch-size | model  | sample_get | move-graph | move-feature | forward | backward  | epoch_time | accuracy | data-moved | edges-processed\n")
    for graphname, hidden_size, batch_size in settings:
        try:
            for cache in cache_per:
                print("Handling cache",cache)
                run_experiment_on_graph(graphname, model,  hidden_size, batch_size, cache)
        except:
            import traceback
            with open("exeption_pagraph",'a') as fp:
                import sys
                ex_type, ex, tb = sys.exc_info()
                traceback.print_exception(ex_type, ex, tb)
                traceback.print_tb( tb, file = fp)
                traceback.print_exception(ex_type, ex, tb, file = fp)

if __name__ == "__main__":
    #run_model("gcn")
    #print("Success!!!!!!!!!!!!!!!!!!!")
    #run_model("gat")
    #return
    import argparse
    argparser = argparse.ArgumentParser("multi-gpu training")
    # Input data arguments parameters.
    argparser.add_argument('--graph',type = str, default= "ogbn-arxiv", required = True)
    # training details
    # model name and details
    argparser.add_argument('--cache-per', type = str,  required = True)
    argparser.add_argument('--model',help="gcn|gat", required = True)
    argparser.add_argument('--batch-size', type=int, required = True)
    # We perform only transductive training
    # argparser.add_argument('--inductive', action='store_false',
    #                        help="Inductive learning setting")
    args = argparser.parse_args()
    dataset = args.graph
    cache_per = args.cache_per
    batch_size = args.batch_size 
    model = args.model
    settings = [(dataset,"16", batch_size)]
    #print(cache_per)
    run_experiment(model, settings, [cache_per])
