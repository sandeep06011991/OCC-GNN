
from exp6 import pagraph
from exp6 import naive
from exp6 import occ

pagraph.run_model('gcn')
pagraph.run_model('gat')
naive.run_experiment_quiver("GAT")
naive.run_experiment_quiver("GCN")
occ.run_experiment_occ("gcn")
occ.run_experiment_occ("gat")
occ.run_experiment_occ("gat-pull")

import exp8

