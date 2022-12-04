
from occ import *
from naive import *

if __name__ == "__main__":
    run_experiment_naive("GCN")
    run_experiment_naive("GAT")

    run_experiment_occ("gcn")
    run_experiment_occ("gat")

