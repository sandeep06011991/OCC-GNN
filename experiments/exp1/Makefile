

single_gpu:
	nvcc -Xcompiler -fopenmp -I ../../src/ ../../src/util/tensor.cu ../../src/gnn/sage.cu  \
		../../src/dataset.cpp single_gpu.cc \
		../../src/util/timer.cpp \
	-lcublas -lnvToolsExt   -o ../../build/sampling
	# nvcc -I ../src/ ../src/tensor.cu ../src/samplers/full.cpp -lcublas \
	# ../src/dataset.cpp gcn_test.cc -o ../build/gcn_test
	../../build/sampling reddit

script:
	python3 run.py pubmed 2
	python3 run.py reddit 2
	../../build/sampling pubmed
	../../build/sampling reddit
