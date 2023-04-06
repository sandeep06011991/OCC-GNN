from baselines.quiver import *
from utils.utils import *

def run_experiment_quiver( model , sample_gpu):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                ("ogbn-arxiv",16, 128, 1024),  \
                ("ogbn-products",16, 100, 1024), \
                ("reorder-papers100M", 16, 128, 1024),\
                ("amazon", 16, 200, 1024),\
                 ]
    no_epochs = 6
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    print("0 Cache has a problem, go over it")
    fanouts = ["10,10,10", "20,20,20", "30,30,30"]
    cache = ".25"
    #settings = [settings[0]]
    #check_path()
    print(settings)
    #sha,dirty = get_git_info()
    layers = 3

    with open('{}/fanouts/quiver.txt'.format(OUT_DIR),'a') as fp:
        #fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  |" + \
            " batch-size | model  | layers | fanout | sample GPU | sample_get | move-data | forward |" +\
              " backward  | epoch_time | accuracy | data_movement | edges \n")
    for graphname, hidden_size, fsize, batch_size in settings:
        for fanout in fanouts:
            out = run_quiver(graphname, model ,no_epochs, cache, hidden_size, fsize, batch_size, layers, fanout, sample_gpu)
            with open('{}/fanouts/quiver.txt'.format(OUT_DIR),'a') as fp:
                fp.write(("{} | {} | {} | {} | {} | {} | {}"+\
                       "| {} | {} | {} | {} | {} | {} | {} | {} |"+\
                       " {} | {}  | {} |  {} \n").format(graphname , "quiver", cache, hidden_size, fsize,\
                        4 * batch_size, model, layers, fanout, sample_gpu,  out["sample_get"], out["movement_data"], \
                        out["movement_feat"], out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                        out["data_moved"], out["edges"]))

if __name__=="__main__":
    run_experiment_quiver("GCN")
    run_experiment_quiver("GAT")
