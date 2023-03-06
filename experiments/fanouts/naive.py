from baselines.naive_gpu import *


def run_experiment_quiver( model ):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                # ("ogbn-arxiv",16, 128, 1024), \
                #("ogbn-arxiv",16, 128, 4096), \
                # ("ogbn-arxiv",16, 128, 256),  \
                # ("ogbn-products",16, 100, 1024), \
                # ("ogbn-products",16, 100, 1024), \
                 ("ogbn-products",16, 100, 4096), \
                 ("ogbn-products",16, 100, 256),  \
                 ("reorder-papers100M", 16, 128,  256),\
                 ("reorder-papers100M", 16, 128, 4096),\
                #("reorder-papers100M", 16, 128, 1024),\
                 ("amazon", 16, 200, 256),\
                 ("amazon", 16, 200,4096),\
                #("amazon", 16, 200, 1024),\
                 ] 
    no_epochs = 5
    # settings = [("ogbn-arxiv",16, 128, 1024)]
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    #cache_rates = [".25"]
    #settings = [settings[0]]
    check_path()
    print(settings)
    sha,dirty = get_git_info()

    with open('{}/fanouts/{}_naive_gpu_sample.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  |" + \
            " batch-size | model  |  layers | fanout |sample_get | move-data | forward |" +\
              " backward  | epoch_time | accuracy | data_movement | edges \n")
    for graphname, hidden_size, fsize, batch_size in settings:
        out = run_quiver(graphname, model ,no_epochs, hidden_size, fsize, batch_size)
        print(out)
        with open('{}/exp6/exp6_{}_naive_gpu_sample.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
            fp.write(("{} | {} | {} | {} | {} "+\
                   "| {} | {} | {} | {} | {} |"+\
                   " {} | {}  | {} | {} \n").format(graphname , SYSTEM , hidden_size, fsize,\
                    4 * batch_size, model, out["sample_get"], out["movement_data_time"] \
                    , out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                    out["data_moved"], out["edges"]))




if __name__=="__main__":
    run_experiment_quiver("GAT")
    run_experiment_quiver("GCN")
