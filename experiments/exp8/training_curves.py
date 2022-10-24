import subprocess
import re
import sys
import os, pwd
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

import os, pwd

uname = pwd.getpwuid(os.getuid())[0]
os.environ['NCCL_BUFFSIZE'] = str(1024 * 1024 * 80)
if uname == 'spolisetty':
    ROOT_DIR = "/home/spolisetty/OCC-GNN"
    SRC_DIR = "/home/spolisetty/OCC-GNN/python/main.py"
    SYSTEM = 'jupiter'
    OUT_DIR = '/home/spolisetty/OCC-GNN/experiments/exp8'
if uname == 'q91':
    ROOT_DIR = "/home/q91/OCC-GNN"
    SRC_DIR = "/home/q91/OCC-GNN/python/main.py"
    SYSTEM = 'ornl'
    OUT_DIR = '/home/q91/OCC-GNN/experiments/exp8'
if uname == 'spolisetty_umass_edu':
    ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN"
    SRC_DIR = "/home/spolisetty_umass_edu/OCC-GNN/python/main.py"
    SYSTEM = 'unity'
    OUT_DIR = '/home/spolisetty_umass_edu/OCC-GNN/experiments/exp8'



def average_string(ls):
    print(ls)
    c = [float(c) for c in ls]
    c = sum(c)/len(c)
    return c


# measures cost of memory transfer of dataset
def run_occ_accuracy(graphname, model, epochs, hidden_size, fsize, minibatch_size, test_dir):
    assert(model in ["gcn", "gat"])
    output = subprocess.run(["python3",\
            "{}/python/main.py".format(ROOT_DIR),\
            "--graph",graphname,  \
            "--model", model , \
            "--num-hidden",  str(hidden_size), \
            "--batch-size", str(minibatch_size * 4), \
                 "--num-epochs", str(epochs) , "--test-graph-dir", test_dir, "--cache-per", ".25"] \
                , capture_output = True)

    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
        # Accuracy: 0.11052815616130829, device:0, epoch:0
        test_accuracy = re.findall("test_accuracy:(\d+\.\d+), epoch:(\d+)", out)
        print("Accuracy read ", test_accuracy)
        epoch_accuracy = {}
        for a,b in test_accuracy:
            epoch_accuracy[int(b)] = a

    except Exception as e:
        with open('exception_occ_acc.txt','a') as fp:
            fp.write(error)
    return epoch_accuracy

# measures cost of memory transfer of dataset
def run_naive_accuracy(graphname, model, epochs, hidden_size, fsize, minibatch_size, test_dir):
    assert(model in ["GCN","GAT"])
    output = subprocess.run(["python3",\
            "{}/quiver/naive_dgl_gcn.py".format(ROOT_DIR),\
            "--graph",graphname,  \
            "--model", model , \
            "--num-hidden",  str(hidden_size), \
            "--batch-size", str(minibatch_size), \
                 "--num-epochs", str(epochs) , "--test-graph", test_dir] \
                , capture_output = True)

    out = str(output.stdout)
    error = str(output.stderr)
    print(out,error)
    #print("Start Capture !!!!!!!", graphname, minibatch_size)
    try:
        # Accuracy: 0.11052815616130829, device:0, epoch:0
        test_accuracy = re.findall("Accuracy: (\d+\.\d+), device:\d, epoch:(\d+)",out)
        print("Accuracy read ", test_accuracy)
        epoch_accuracy = {}
        for a,b in test_accuracy:
            epoch_accuracy[int(b)] = a

    except Exception as e:
        with open('exception_naive_acc.txt','a') as fp:
            fp.write(error)
    return epoch_accuracy


def run_accuracy( model , system):
    assert(system in ["occ", "naive"])
    # graph, hidden_size, fsize, training dir
    settings = [
                ("ogbn-products",16, 128, "ogbn-products"), \
                 ]
    no_epochs = 8
    mode = []
    check_path()
    print(settings)
    sha,dirty = get_git_info()
    batch_size = 1024
    with open('{}/exp7_accuracy.txt'.format(OUT_DIR),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph  | model | system |  hidden | epochs \n")
    for graphname, hidden_size, fsize, test_dir in settings:
        if system == "naive":
            out = run_naive_accuracy(graphname, model , no_epochs, hidden_size, fsize, batch_size, test_dir)
        else:
            out = run_occ_accuracy(graphname, model , no_epochs, hidden_size, fsize, batch_size, test_dir)
        str = ""
        vals = []
        for k in out.keys():
            str += "| {} "
            vals.append(out[k])
        vals = tuple(vals)
        with open('{}/exp7_accuracy.txt'.format(OUT_DIR),'a') as fp:
            fp.write(("{} | {} | {} | {} | {} " + str + "\n").format(graphname , model , system,  hidden_size, no_epochs, *vals))




if __name__=="__main__":
    run_accuracy("GCN", "naive")
    run_accuracy("gcn","occ")
    # print(test_accuracy[0])
    # run_experiment_quiver("GAT")
    # run_experiment_quiver("GCN")
