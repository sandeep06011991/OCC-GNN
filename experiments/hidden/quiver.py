from baselines.quiver import *


def run_experiment_quiver( model, cache ):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv", 128,  1024), \
                ("ogbn-products", 100, 1024), \
                ("reorder-papers100M", 128,  1024),\
                #("amazon", 200, 1024)
                 ]
    no_epochs = 6
    print("0 Cache has a problem, go over it")
    fanout = "20,20,20"
    #cache = ".25"
    #settings = [settings[0]]
    #check_path()
    print(settings)
    #sha,dirty = get_git_info()
    layers = 3

    hidden_sizes = [16, 64, 128]
    #hidden_sizes = [16]
    with open('{}/hidden/quiver.txt'.format(OUT_DIR),'a') as fp:
        #fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  |" + \
            " batch-size | model  | layers | fanout |  sample_get | move-data | forward |" +\
              " backward  | epoch_time | accuracy | data_movement | edges \n")
    for graphname, fsize, batch_size in settings:
        for hidden_size in hidden_sizes:
            out = run_quiver(graphname, model ,no_epochs, cache, hidden_size, fsize, batch_size, layers, fanout)
            with open('{}/hidden/quiver.txt'.format(OUT_DIR),'a') as fp:
                fp.write(("{} | {} | {} | {} | {} | {} "+\
                       "| {} | {} | {} | {} | {} | {} | {} | {} |"+\
                       " {} | {}  | {} |  {} \n").format(graphname , "quiver", cache, hidden_size, fsize,\
                        4 * batch_size, model, layers, fanout, out["sample_get"], out["movement_data"], \
                        out["movement_feat"], out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                        out["data_moved"], out["edges"]))

if __name__=="__main__":
    #run_experiment_quiver("GCN",".25")
    run_experiment_quiver("GAT", ".25")
    run_experiment_quiver("GCN", ".10")
    run_experiment_quiver("GAT", ".10")
