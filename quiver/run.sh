# Overnight scripts
#python3 quiver_train.py --model GCN --batch-size 1024 \
#    --graph ogbn-papers100M --num-hidden 256 --cache-size 7GB 
python3 dgl_train.py --model GAT --batch-size 1024 \
	--graph ogbn-arxiv --num-hidden 256

# python3 dgl_train.py --model GAT --batch-size 1024 \
# 	--graph ogbn-papers100M --num-hidden 256 

#python3 quiver_train.py --model GAT --batch-size 1024 \
 #   --graph ogbn-papers100M --num-hidden 256 --cache-size 4GB

#python3 gpu_sample_naive_dgl_gcn.py --model GCN --batch-size 4096  --graph ogbn-arxiv  --num-hidden 256
#nsys profile --trace-fork-before-exec true -o qv_pa-gat.nsys-rep --force-overwrite true -c cudaProfilerApi \
#	python3 dgl_gcn.py --model GAT --batch-size 1024 --cache-per .25 --graph reorder-papers100M --num-hidden 512 --sample-gpu
# python3 naive_dgl_gcn.py --model GAT --batch-size 1024 --graph reorder-papers100M --num-hidden 256 --sample-gpu 
    
