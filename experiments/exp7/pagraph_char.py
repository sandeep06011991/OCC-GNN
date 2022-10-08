import torch
from cslicer import stats
import os, pwd

uname = pwd.getpwuid(os.getuid())[0]

if uname == 'spolisetty':
    ROOT_DIR = "/home/spolisetty/OCC-GNN/cslicer/"
    SRC_DIR = "/home/spolisetty/OCC-GNN/python/main.py"
    SYSTEM = 'jupiter'
    OUT_DIR = '/home/spolisetty/OCC-GNN/experiments/exp7'
if uname == 'q91':
    ROOT_DIR = "/home/q91/OCC-GNN/cslicer/"
    SRC_DIR = "/home/q91/OCC-GNN/python/main.py"
    SYSTEM = 'ornl'
    OUT_DIR = '/home/q91/OCC-GNN/experiments/exp7'
if uname == 'spolisetty_umass_edu':
    ROOT_DIR = "/home/spolisetty_umass_edu/OCC-GNN/cslicer"
    SRC_DIR = "/home/spolisetty_umass_edu/OCC-GNN/python/main.py"
    SYSTEM = 'unity'
    OUT_DIR = '/home/spolisetty_umass_edu/OCC-GNN/experiments/exp7'

def get_data_dir():
    import os
    username = os.environ['USER']
    if username == 'spolisetty_umass_edu':
        DATA_DIR = "/work/spolisetty_umass_edu/data"
    if username == "spolisetty":
        DATA_DIR = "/data/sandeep"
    if username == "q91":
        DATA_DIR = "/mnt/bigdata/sandeep"
    return DATA_DIR

DATA_DIR = get_data_dir()


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

# Load training.
def run_experiment(graphname, batch_size, partition_config):
    # Read partition map
    partition_map = load
    num_epochs = 5
    if(graphname == "com-orkut"):
        print("warning not done !!")
    assert(graphname != "com-orkut")
    idx_mask = np.fromfile(
            "{}/{}/{}_idx.bin".format(DATA_DIR, graphname, train), dtype=np.int64)
    train_nids = torch.from_numpy(idx_mask)
    s = stats(graphname, "occ", 10)
    d = {}
    d["redundant_communication"] = []
    d["total_communication"] = []
    d["redundant_computation"] = []
    d["total_computation"] = []
    for _ in range(num_epochs):
        random.shuffle(training_nodes)
        i = 0
        redundant_computation = 0
        total_computation = 0
        redundant_communication = 0
        total_computation  = 0
        while(i < num_nodes):
            sample = (training_nodes[i:i+batch_size])
            t = s.get_stats(sample.list())
            # Decide which is better for redundant commutation 
            i = i + batch_size


def run_experiment():
    graphs = ["ogbn-arxiv","ogbn-products"]
    graphs = ["ogbn-arxiv"]
    batch_size = [4096, 4096 * 4]
    partition = ["pagraph", "random"]
    partition = ["random"]
    with open("{}/{}.txt".format(OUT_DIR, SYSTEM),'a') as fp:
        fp.write("graph_name | batch_size | partitioning | per-computation | per-communication \n");

    check_path()
    for g in graphs:
        for b in batch_size:
            d = run_experiment(g, batch_size, partition_config)
            with open("{}/{}.txt".format(OUT_DIR, SYSTEM),'a') as fp:
                fp.write("graph_name | batch_size | partitioning | per-computation | per-communication \n");

if __name__=="__main__":
    print("hello world")
