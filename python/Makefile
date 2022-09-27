all:
	cd ../cslicer; python3 setup.py build_ext --inplace
	python3 main.py --graph ogbn-arxiv --num-epochs 1 --cache-per 1 --model gcn
