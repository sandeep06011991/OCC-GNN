from occ import run_experiment_occ
from pagraph import run_experiment_pagraph

if __name__ == "__main__":
    settings = [\
               ("ogbn-arxiv",3, 32, -1, 4096),\
               ("ogbn-arxiv",3, 256, -1, 4096),\
               ("ogbn-arxiv",3, 32 , -1 , 1024),\
               ("ogbn-products",3, 32, -1, 4096),\
               ("ogbn-products",3, 256, -1, 4096),\
               ("ogbn-products",3, 32 , -1 , 1024),\
               ("com-youtube", 3, 32, 256, 4096),\
               ("com-youtube",3,32 ,1024, 4096),\
               #  #("ogbn-products",2), \
                ("ogbn-papers100M",2, 256,-1,4096), \
                ("ogbn-papers100M",2, 32,-1,4096), \
                # ("com-friendster",2), \
                # ("com-orkut",5, 256 , 256, 4096 ) \
                ]
    for ss in settings:
        print(ss)
        run_experiment_occ([ss])
        # run_experiment_pagraph([ss])
