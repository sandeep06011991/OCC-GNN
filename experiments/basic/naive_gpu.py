from baselines.naive_gpu import *

def run_naive_experiment( model, sample_gpu):
    # graph, hidden_size, fsize, minibatch_size
    settings = [
                #("ogbn-arxiv", 128,  1024), \
                #("ogbn-products", 100, 1024 ), \
                #("train_reorder_papers100M", 128, 1024),\
                #("ogbn-papers100M", 128 , 1024),\
                # ("train_reorder_mag240M", 768, 1024),\
                 ("mag240M", 768, 1024),\
                # ("reorder-mag240M", 768, 1024),\
                #("amazon", 200, 1024),\
               # ("ogbn-products", 100, 256), \
               # ("reorder-papers100M", 128, 256),\
               # ("amazon", 200, 256),\
                 ]
    no_epochs = 6
    print("0 Cache has a problem, go over it")
    fanout = "20,20,20"
    #settings = [settings[0]]
    #check_path()
    print(settings)
    #sha,dirty = get_git_info()
    layers = 3
    hidden_sizes = [ 256]
    with open('{}/basic/naive.txt'.format(OUT_DIR),'a') as fp:
        #fp.write("sha:{}, dirty:{}\n".format(sha,dirty))
        fp.write("graph,system,hidden-size,fsize," + \
            "batch-size,model,layers,fanout,mode, sample_GPU,move-data,forward," +\
              "backward,epoch_time,accuracy,data_movement,edges,memory(GB) \n")
    for graphname, fsize, batch_size in settings:
        for hidden_size in hidden_sizes:
            if sample_gpu: 
                MODE = "GPU"
            else: 
                MODE = "UVA"    
            print("Check")
            out = run_naive(graphname, model ,no_epochs, hidden_size, fsize, batch_size,  sample_gpu)
            with open('{}/basic/naive.txt'.format(OUT_DIR),'a') as fp:
                fp.write(("{},{},{},{},{},{},{},{},"+\
                       "{},{},{},{},{},{},{},{},"+\
                       "{},{}\n").format(graphname , "Naive",  hidden_size, fsize,\
                        4 * batch_size, model, layers, fanout.replace(",","-"), MODE,  out["sample_get"], out["movement_data_time"], \
                         out["forward"], out["backward"],  out["epoch"], out["accuracy"],
                        out["data_moved"], out["edges"], out["memory_used"]))

if __name__=="__main__":
    for sample_gpu in [False]:
        run_naive_experiment("GCN",sample_gpu)
        run_naive_experiment("GAT", sample_gpu) 
