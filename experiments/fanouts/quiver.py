from baselines import *


def run_experiment_quiver( model ):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv",16, 128, 1024), \
                #("ogbn-arxiv",16, 128, 4096), \
                #("ogbn-arxiv",16, 128, 256),  \
                ("ogbn-products",16, 100, 1024), \
                #("ogbn-products",16, 100, 4096), \
                #("ogbn-products",16, 100, 256),  \
                #("reorder-papers100M", 16, 256),\
                #("reorder-papers100M", 16, 4096),\
                #("reorder-papers100M", 16, 1024),\
                #("com-youtube", 3, 32, 256, 4096),\
                #("com-youtube",3,32,1024, 4096)\
                # ("com-youtube",2), \
                # ("ogbn-products",2), \
                # ("ogbn-papers100M",2), \
                # ("com-friendster",2), \
                 # ("com-orkut",5, 256, 256, 4096) \
                 ]
    no_epochs = 6
    # settings = [("ogbn-papers100M",2)]
    # cache_rates = [".05",".10",".24",".5"]
    # cache_rates = [".05",".24", ".5"]
    print("0 Cache has a problem, go over it")
    fanouts = ["10,10,10"]
    cache = ".25"
    #settings = [settings[0]]
    #check_path()
    print(settings)
    sha,dirty = get_git_info()
    layers = 3

    with open('{}/fanouts/quiver.txt'.format(OUT_DIR),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize  |" + \
            " batch-size | model  | layers | fanout |  sample_get | move-data | forward |" +\
              " backward  | epoch_time | accuracy | data_movement | edges \n")
    for graphname, hidden_size, fsize, batch_size in settings:
        for fanout in fanouts:
            out = run_quiver(graphname, model ,no_epochs, cache, hidden_size, fsize, batch_size)
            with open('{}/fanouts/quiver.txt'.format(OUT_DIR),'a') as fp:
                fp.write(("{} | {} | {} | {} | {} | {} "+\
                       "| {} | {} | {} | {} | {} | {} | {} | {} |"+\
                       " {} | {}  | {} |  {} \n").format(graphname , "quiver", cache, hidden_size, fsize,\
                        4 * batch_size, model, layers, fanout, out["sample_get"], out["movement_data"], \
                        out["movement_feat"], out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                        out["data_moved"], out["edges"]))
