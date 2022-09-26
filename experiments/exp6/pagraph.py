 # Ensure cuda/10.2 is loaded
import subprocess
import os
import time

ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/pagraph"
import sys
import re
import git

Feat = {"ogbn-arxiv":"128","ogbn-products":"100", "ogbn-papers100M":"128"}

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

def start_client(filename):
    feat_size = Feat[filename]
    cmd = ['python3','{}/examples/profile/pa_gcn.py'.format(ROOT_DIR), '--dataset', filename,'--n-epochs','2'\
                    ,'--feat-size',feat_size]
    output = subprocess.run(cmd, capture_output=True)
    out = str(output.stdout)
    err = str(output.stderr)
    print(out,err)
    output = out
    sample = float(re.findall("Sample time: (\d+\.\d+)s",output)[0])
    compute  = float(re.findall("Compute time: (\d+\.\d+)s",output)[0])
    collect  = float(re.findall("CPU collect: (\d+\.\d+)s",output)[0])
    move  = float(re.findall("CUDA move: (\d+\.\d+)s",output)[0])
    epoch  = float(re.findall("Epoch time: (\d+\.\d+)s",output)[0])
    miss_rate = float(re.findall("Miss rate: (\d+\.\d+)s",output)[0])

    return {"sample":sample, "compute":compute, "collect":collect,\
            "move":move, "epoch":epoch, "miss_rate": miss_rate}

def run_experiment_on_graph(filename):
    fp = start_server(filename)
    res = start_client(filename)
    WRITE = "{}/experiments/exp1.txt".format(ROOT_DIR)

    with open(WRITE,'a') as fp:
        fp.write("{}|{}|{}|{}|{}|{}|{}\n".format(filename,res["sample"], res["collect"],res["move"], res["compute"], \
                        res["epoch"], res["miss_rate"]))
    fp.close()

def run_experiment():
    graphs = ['ogbn-arxiv','ogbn-products']
    #graphs = ['ogbn-papers100M']
    #graphs = ['ogbn-arxiv']
    sha, dirty = get_git_info()
    check_path()
    check_no_stale()
    filename = "{}/experiments/exp1.txt".format(ROOT_DIR)
    with open(filename,'a') as fp:
        fp.write("Git hash:{}, Dirty:{}\n".format(sha, dirty))
        fp.write("File |  Sample | Collect | Move | Compute | Epoch | Miss rate \n")
    for i in graphs:
        try:
            run_experiment_on_graph(i)
        except:
            import traceback
            with open("exeption",'a') as fp:
                import sys
                ex_type, ex, tb = sys.exc_info()
                traceback.print_exception(ex_type, ex, tb)
                traceback.print_tb( tb, file = fp)
                traceback.print_exception(ex_type, ex, tb, file = fp)

if __name__ == "__main__":
    run_experiment()
