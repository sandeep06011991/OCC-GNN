from baselines.quiver import *


def run_experiment_quiver( model, cache, sample_gpu):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv", 128,  1024), \
                #("ogbn-products", 100, 1024 ), \
                ("ogbn-papers100M", 128, 1024),\
                ("mag240M", 768, 1024),\
                #("reorder-papers100M", 128, 1024),\
                # ("reorder-mag240M", 768, 1024),\
                #("amazon", 200, 1024),\
               # ("ogbn-products", 100, 256), \
               # ("reorder-papers100M", 128, 256),\
               # ("amazon", 200, 256),\
                 ]
    no_epochs = 6
    print("0 Cache has a problem, go over it")
    fanout = "20,20,20"
    #sha,dirty = get_git_info()
    layers = 3

    hidden_sizes = [ 256]
    with open('{}/basic/quiver.txt'.format(OUT_DIR),'a') as fp:
        #fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph,system,cache,hidden-size,fsize," + \
            "batch-size,model,layers,fanout,MODE,sample_get,move-data,forward," +\
              "backward,epoch_time,accuracy,data_movement,edges,memory(GB) \n")
    for graphname, fsize, batch_size in settings:
        for hidden_size in hidden_sizes:
            out = run_quiver(graphname, model ,no_epochs, cache, hidden_size, fsize, batch_size, layers, fanout, sample_gpu)
            with open('{}/basic/quiver.txt'.format(OUT_DIR),'a') as fp:
                if sample_gpu: 
                    MODE = "GPU"
                else:
                    MODE = "UVA"

                fp.write(("{},{},{},{},{},{},{},"+\
                       "{},{},{},{},{},{},{},{},"+\
                       "{},{},{},{}\n").format(graphname , "quiver", cache, hidden_size, fsize,\
                        4 * batch_size, model, layers, fanout.replace(",","-"), MODE, out["sample_get"], out["movement_data"], \
                         out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                        out["data_moved"], out["edges"], out["memory_used"]))

if __name__=="__main__":

    for sample_gpu in [ False, True]:
        for cache_per in [".05", ".10", ".15", "20"]:
            run_experiment_quiver("GAT",cache_per , sample_gpu)
            run_experiment_quiver("GCN",cache_per , sample_gpu)
        #run_experiment_quiver("GAT", ".10", sample_gpu)
    #sample_gpu = False
    #run_experiment_quiver("GAT", ".25", sample_gpu)
    #run_experiment_quiver("GCN", ".10", sample_gpu)
    #run_experiment_quiver("GAT", ".10", sample_gpu)
