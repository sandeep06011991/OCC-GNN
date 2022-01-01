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
     # python3 server/pa_server.py --dataset ~/data/pagraph/lj/ --gnn-layers 4 --num-neighbors 10
# 4. python3 examples/profile/pa_gcn.py --dataset ~/data/pagraph/lj/ --n-layers 4 --gpu 0,1,2,3 --batch-size 256

3. Run the dgl gcn example with the limited graph size

def run_graph_store_server():
    pass
def run_example():
    pass

def run_experiment(graph_name, fsize, hop, max_memory):


def populate_table():
    with open("exp3.txt",'a') as fp:
        fp.write("Graph | fsize | hop | Max-Memory | Cache-hit rate \n")
    run_experiment()


if __main__ == "__name__":
    populate_table()
