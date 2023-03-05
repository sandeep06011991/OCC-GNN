from baselines.pagraph import *

def run_model(model):
    graphs = ['ogbn-arxiv','ogbn-products']

    #graphs = ['ogbn-papers100M']
    #graphs = ['ogbn-arxiv']
    settings = [#('ogbn-arxiv', 16, 1024),
                ('ogbn-arxiv', 16, 4096),
                # ('ogbn-products', 16, 4096),
                # ('reorder-papers100M',16, 4096),
                # ('amazon', 16, 4096),
                # ('amazon', 16, 256)
                ]

    cache_per = ".25"
    num_layers = 3
    fanouts = ["10,10,10", "20,20,20","30,30,30"]
    #cache_per = [ "0",".25"]
    run_experiment(model, settings, cache_per, fanouts, num_layers)

    #settings = [('ogbn-arxiv',16,1024)]
def run_experiment(model, settings, cache_per, fanouts, num_layers):
    sha, dirty = get_git_info()
    check_path()
    check_no_stale()
    with open('{}/fanouts/{}_pagraph.txt'.format(OUT_DIR, SYSTEM),'a') as fp:
        fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph | system | cache |  hidden-size | fsize " +\
            "| batch-size | model  | layers | fanout | sample_get | data-moved | forward | backward  | epoch_time | accuracy | data-moved | edges-processed\n")
    for graphname, hidden_size, batch_size in settings:
        try:
            feat_size = Feat[graphname]
            for fanout in fanouts:
                res = run_experiment_on_graph(graphname, model,  hidden_size, batch_size, cache_per,\
                 fanout, num_layers)
                WRITE = '{}/fanouts/{}_pagraph.txt'.format(OUT_DIR, SYSTEM)
                with open(WRITE,'a') as fp:

                    fp.write("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(graphname, "jupiter", cache_per, hidden_size, feat_size, \
                            4 * batch_size, model, num_layers, fanout, res["sample"], res["total_movement"],res["forward"], \
                                res["backward"],res["epoch_time"], res["accuracy"],res["miss_num"], res["edges"] ))
                fp.close()
        except:
            import traceback
            with open("exeption_pagraph",'a') as fp:
                import sys
                ex_type, ex, tb = sys.exc_info()
                traceback.print_exception(ex_type, ex, tb)
                traceback.print_tb( tb, file = fp)
                traceback.print_exception(ex_type, ex, tb, file = fp)

if __name__ == "__main__":
    run_model("gcn")
    #print("Success!!!!!!!!!!!!!!!!!!!")
    # run_model("gat")
