'''
Creates table
Graph | Hidden_Size | Model | batch_size | cache |
    Quiver-GPU(Sample) | Quiver(GPU-Load) | Quiver-GPU (Train) | Quiver-GPU(Epoch)'''

import pandas

def check_non(series):
    if(len(series) == 1):
        return
    print(series)
    assert(False)

def write_table(cache):
    quiver = pandas.read_csv('quiver.txt', sep = '|', header = 0, index_col = False)
    groot_cpu = pandas.read_csv('occ_cpuP4.txt', sep = '|', header = 0, index_col =  False)
    groot_gpu = pandas.read_csv('occ_P4.txt', sep = '|', header = 0, index_col = False)


    graphs = ['ogbn-products ', 'reorder-papers100M ', 'amazon ']
    batches = [1024 , 4096]
    hidden_sizes = [ 16 , 64 ]
    #caches = [.1, .25]
    #with open('relative.txt', 'w') as fp:
    #    fp.write("graph | hidden | model | batch | cache | Quiver-GPU(Sample)" +\
    #            "|Quiver-GPU(Load)|Quiver-GPU(Train)|Quiver-GPU(Epoch)|Quiver-CPU(Load)|" +\
    #            "Quiver-CPU(Train)|GRoot-GPU(Sample)|GRoot-GPU(Train)|GRoot-CPU(Train)\n")

    for graph in graphs:
        for batch in batches:
            for hidden in hidden_sizes:
                for model in [" gcn ", " gat "]:
                    model_q = model.upper()
                    if model == " gat ":
                        model_cpu = " gat-pull "
                    else:
                        model_cpu = model
                    _quiver = (quiver['graph '] == graph) & (quiver[' model  '] == model_q) \
                           & (quiver[' batch-size '] == batch) & (quiver['  hidden-size '] == hidden) & (quiver[' cache '] == cache)
                    
                    _groot_cpu = groot_cpu[(groot_cpu['graph '] == graph)   & (groot_cpu[' model  '] == model_cpu) & \
                                (groot_cpu[' batch-size '] == batch) & (groot_cpu['  hidden-size '] == hidden) & (groot_cpu[' cache '] == cache)]
                    
                    _groot_gpu =  groot_gpu[(groot_gpu['graph '] == graph)  & (groot_gpu[' model  '] == model) &\
                         (groot_gpu[' batch-size '] == batch) & (groot_gpu['  hidden-size '] == hidden) & (groot_gpu[' cache '] == cache)]
                    
                    quiver_gpus = quiver[ _quiver & (quiver[' sample_GPU '] == " True ")]
                    quiver_cpus = quiver[ _quiver & (quiver[' sample_GPU '] == " False ")]
                    #print(quiver_gpus, quiver_cpus, _groot_gpu, _groot_cpu)
                    if(len(_groot_gpu)>1):
                        print(_groot_gpu)
                    if(len(_groot_cpu)>1):
                        print(_groot_cpu)
                    if(len(quiver_gpus) >1):
                        print(quiver_gpus)
                    if(len(quiver_cpus) > 1):
                        print(quiver_cpus)
                    if(len(quiver_gpus) == 1):
                        quiver_gpu_sample = quiver_gpus[" sample_get "].iloc[0]
                        quiver_gpu_load = quiver_gpus[" move-data "].iloc[0]
                        quiver_gpu_train = quiver_gpus[" forward "].iloc[0] + quiver_gpus[" backward  "].iloc[0]
                        quiver_gpu_epoch = quiver_gpus[" epoch_time "].iloc[0]
                    else:
                        quiver_gpu_sample = "error"
                        quiver_gpu_load = "error"
                        quiver_gpu_train = "error"
                        quiver_gpu_epoch = "error"
                    if (len(quiver_cpus) == 1):
                        quiver_cpu_load = quiver_cpus[" move-data "].iloc[0]
                        quiver_cpu_train = quiver_cpus[" forward "].iloc[0] + quiver_cpus[" backward  "].iloc[0]
                    else:
                        quiver_cpu_load = "error"
                        quiver_cpu_train = "error"
                    if(len(_groot_gpu) == 1):
                        groot_gpu_sample = _groot_gpu["  sample_get "].iloc[0]
                        groot_gpu_train = _groot_gpu[" forward "].iloc[0] + _groot_gpu[" backward  "].iloc[0]
                        groot_gpu_epoch = _groot_gpu[" epoch_time "].iloc[0]
                    else:
                        groot_gpu_sample = "error"
                        groot_gpu_train = "error"
                        groot_gpu_epoch = "error"
                    if(len(_groot_cpu) == 1):
                        groot_cpu_train = _groot_cpu[" forward "].iloc[0] + _groot_gpu[" backward  "].iloc[0]
                    else:
                        groot_cpu_train = "error"

                    with open('relative.txt', 'a') as fp:
                        fp.write(("{}|{}|{}|{}|{}|{}|{}|" + \
                                "{}|{}|{}|{}|{}|{}|{}|{}\n").format(
                            graph, hidden, model, batch, cache,  quiver_gpu_sample, quiver_gpu_load,\
                                quiver_gpu_train, quiver_gpu_epoch, quiver_cpu_load, quiver_cpu_train,\
                                groot_gpu_sample, groot_gpu_train, groot_gpu_epoch,  groot_cpu_train))
                        
                        
        



if __name__ == "__main__":
    with open('relative.txt', 'w') as fp:
        fp.write("graph | hidden | model | batch | cache | Quiver-GPU(Sample)" +\
                "|Quiver-GPU(Load)|Quiver-GPU(Train)|Quiver-GPU(Epoch)|Quiver-CPU(Load)|" +\
                "Quiver-CPU(Train)|GRoot-GPU(Sample)|GRoot-GPU(Train)|Groot-GPU(Epoch)| GRoot-CPU(Train)\n")

    write_table(.25)
    write_table(.1)
