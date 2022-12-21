
from exp6 import pagraph
from exp6 import naive
from exp6 import occ
from exp6 import quiver_exp

#pagraph.run_model('gcn')
#pagraph.run_model('gat')
naive.run_experiment_quiver("GAT")
naive.run_experiment_quiver("GCN")
#occ.run_experiment_occ("gcn")
#occ.run_experiment_occ("gat")
#occ.run_experiment_occ("gat-pull")

quiver_exp.run_experiment_quiver("GCN")
print("All gcn done")
quiver_exp.run_experiment_quiver("GAT")
print("All GAT done")


#from exp8 import naive
#from exp8 import occ
#
#naive.run_experiment_naive("GCN")
#naive.run_experiment_naive("GAT")
#occ.run_experiment_occ("gcn")
#occ.run_experiment_occ("gat")

#import exp8
