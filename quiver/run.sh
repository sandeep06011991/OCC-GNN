python3 dgl_gcn.py --model GAT  --batch-size 1024 --cache-per 0.05  --graph mag240M --num-hidden 256 

#python3 gpu_sample_naive_dgl_gcn.py --model GCN --batch-size 4096  --graph ogbn-arxiv  --num-hidden 256
#nsys profile --trace-fork-before-exec true -o qv_pa-gat.nsys-rep --force-overwrite true -c cudaProfilerApi \
#	python3 dgl_gcn.py --model GAT --batch-size 1024 --cache-per .25 --graph reorder-papers100M --num-hidden 512 --sample-gpu

# python3 naive_dgl_gcn.py --model GAT --batch-size 1024 --graph reorder-papers100M --num-hidden 256 --sample-gpu 
    
