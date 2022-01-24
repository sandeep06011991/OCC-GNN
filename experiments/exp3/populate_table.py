# Description of the experiment
# Goal: Measure cache hit rate for varying graph data sizes and K-hop depth
# Prove that with increasing depth cache utilization is not scalable
# We keep Fsize flexible as that is the dimension used by P3 in parallelizing.
# Measure cache hit rate for higher depth and large graphs.
# Main trend lines I am looking for
# 3 comparable large graphs, 2-5-7 hops, cache hit rate.
# Constrain the gpu as finding graphs larger than 24 GB is a bit painfull to preprocessself.
# Using fsize as another dimension to work on for P3
# Graph | Fsize |  Hop | Max-Memory | Cache-hit rate
# For all datasets
# 1. Run the preprocessor to convert point to point graph text file into npz format
    # python3 PaGraph/data/preprocess.py --dataset ~/data/pagraph/lj --ppfile com-lj.ungraph.txt --gen-feature --gen-label --gen-set
# 2. Generate sub partitions for each of the graphs
    # python3 PaGraph/partition/metis.py --dataset ~/data/pagraph/lj --partition 4 --num-hops 3
# 3. Python run graph store server
     # python3 server/pa_server.py --dataset ~/data/pagraph/lj/ --gnn-layers 4 --num-neighbors 2
# 4. python3 examples/profile/pa_gcn.py --dataset ~/data/pagraph/lj/ --n-layers 4 --gpu 0,1,2,3 --batch-size 256

# 3. Run the dgl gcn example with the limited graph size
import os
os.environ["PYTHONPATH"] = "/home/spolisetty/OCC-GNN/external/PaGraph/"
import re
import sys
from subprocess import PIPE, Popen
from threading  import Thread
import time
import subprocess

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # python 2.x

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    # print("REaching here")
    try:
        for line in iter(out.readline,""):
            # print(line,"MARK")
            if "start running graph server" in line :
                print("Graph server alert")
                queue.put(line)
    except:
        return


def run_graph_store_server(dataset):
    p = Popen(["python3","-u","/home/spolisetty/OCC-GNN/external/PaGraph/server/pa_server.py",
            "--dataset","/home/spolisetty/data/pagraph/{}".format(dataset),
            "--gnn-layers","4","--num-neighbors","10"], stdout=PIPE, \
            bufsize=1, close_fds=ON_POSIX,universal_newlines=True)
    q = Queue()
    t = Thread(target = enqueue_output, args = (p.stdout,q))
    t.start()
    while True:
        try:
            line = q.get_nowait()
            # print("recieved line",line)
            if "running" in line:
                print("now  server is up")
                break
        except Empty:
            pass
        time.sleep(1)
    return p,t


def run_experiment(graph_name,cache_per):
    p = None
    try:
        p,t = run_graph_store_server(graph_name)
        output = subprocess.run(["python3","/home/spolisetty/OCC-GNN/external/PaGraph/examples/profile/pa_gcn.py",
                "--dataset","/home/spolisetty/data/pagraph/{}".format(graph_name),
                "--n-layers","4","--num-neighbors","10","--cache-percentage",str(cache_per),"--gpu","0,1,2,3","--n-epochs","1"], capture_output=True)
        print(output,"out")
        output = str(output.stdout)
        print(output)
        print("Exiting")
        miss_rate = re.findall(r"average miss rate:\ (\d+\.\d+)",output)[0]
        epoch_time = re.findall(r"average epoch time:\ (\d+\.\d+s)",output)[0]
        print("found",miss_rate,epoch_time)
        print(p)
        p.terminate()
        # p.stdout.close()
        # t.join()
        return miss_rate, epoch_time
    except Exception as e:
        print(e)
        if p != None:
            p.terminate()
        return "OOM","OOM"
    pass

def populate_table():
    graph_names = ["reddit","lj"]
    cache_percentage = [0,10,25,100]
    graph_name = ["lj","reddit"]
    # cache_percentage = [1,100]
    with open("exp3.txt",'a') as fp:
        fp.write("Graph | cache-percentage | cache-miss rate | avg-epoch time \n")
    for g in graph_name:
        for c in cache_percentage:
            (miss_rate, epoch_time) = run_experiment(g,c)
            with open("exp3.txt",'a') as fp:
                fp.write("{}|{}|{}|{} \n".format(g,c,miss_rate, epoch_time))



if __name__ == "__main__":
    populate_table()
