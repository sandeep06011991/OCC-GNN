python3 main.py --model gat --batch-size 4096 --cache-per .25 --graph reorder-papers100M --num-gpus 4 --optimization1  --num-hidden 64

#nsys profile --trace-fork-before-exec true -o pr-gat.nsys-rep --force-overwrite true -c cudaProfilerApi \
#	python3 main.py --model gat --batch-size 4096 --cache-per .25 --graph ogbn-products --num-gpus 4 --optimization1 --skip-shuffle --num-hidden 64
