python3 main.py --model gat --batch-size 4096 --cache-size 2MB  --graph ogbn-arxiv --num-gpus 4 --use-uva  --num-hidden 256


#nsys profile --trace-fork-before-exec true -o groot-pa-gat-bal.nsys-rep --force-overwrite true -c cudaProfilerApi \
#	python3 main.py --model gat --batch-size 4096 --cache-per 0 --graph reorder-papers100M --num-gpus 4 --optimization1 --skip-shuffle --num-hidden 256 --load-balance
