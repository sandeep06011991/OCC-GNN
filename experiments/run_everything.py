
#from exp6 import pagraph
#from exp6 import naive
#from exp6 import occ

#pagraph.run_model('gcn')
#pagraph.run_model('gat')
#naive.run_experiment_quiver("GAT")
#naive.run_experiment_quiver("GCN")
#occ.run_experiment_occ("gcn")
#occ.run_experiment_occ("gat")
#occ.run_experiment_occ("gat-pull")

#import exp8
import fanouts.quiver
def setting(sample_gpu):
    #fanouts.quiver.run_experiment_quiver("GCN", sample_gpu)
    #fanouts.quiver.run_experiment_quiver("GAT", sample_gpu)


    import hidden.quiver
    hidden.quiver.run_experiment_quiver("GCN",".25", sample_gpu)
    hidden.quiver.run_experiment_quiver("GAT", ".25", sample_gpu)
    #hidden.quiver.run_experiment_quiver("GCN", ".10", sample_gpu)
    #hidden.quiver.run_experiment_quiver("GAT", ".10", sample_gpu)

    
    #import depth.quiver
    #depth.quiver.run_experiment_quiver("GCN",".25", sample_gpu)
    #depth.quiver.run_experiment_quiver("GAT",".25", sample_gpu)

   
def run_occ():
    import hidden.occ
    hidden.occ.run_experiment_occ("gcn",  False)
    hidden.occ.run_experiment_occ("gat",  False)
    import hidden.occ_cpu
    hidden.occ_cpu.run_experiment_occ("gat-pull", False)
    #hidden.occ.run_experiment_occ("gcn", False)
    #hidden.occ.run_experiment_occ("gat", False)


if __name__=="__main__":
    run_occ()
    #sample_gpu = True
    #setting(sample_gpu)
    #setting(False)
    print("All done")
                                          

                                         
