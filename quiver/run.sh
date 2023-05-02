python3 dgl_gcn.py --model GAT  --batch-size 1024 --cache-per .25 --graph ogbn-products --num-hidden 64 --sample-gpu

#nsys profile --trace-fork-before-exec true -o qv_pr-gat.nsys-rep --force-overwrite true -c cudaProfilerApi \
#	python3 dgl_gcn.py --model GAT --batch-size 1024 --cache-per .25 --graph ogbn-products --num-hidden 64 --sample-gpu
