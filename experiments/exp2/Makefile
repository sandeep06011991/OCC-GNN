# dist:
# 	nvcc -I ../../src/ distributed_gcn.cpp ../../src/util/dist_tensor.cu -l cublas  \
# 	../../src/gnn/dist_sage.cu  ../../src/gnn/sage.cu	\
# 	../../src/util/tensor.cu  ../../src/dataset.cpp \
# 	../../src/util/timer.cpp \
# 	../../src/util/gpu.cu -o ../../build/distributed_gcn
# 	../../build/distributed_gcn pubmed random

naive:
	nvcc -I ../../src/ naive_distributed_gcn.cpp ../../src/util/tensor.cu -l cublas  \
	../../src/gnn/sage.cu  	\
	../../src/dataset.cpp ../../src/util/gpu.cu \
	../../src/util/timer.cpp -lnvToolsExt -o ../../build/naive_distributed_gcn
	# ../../build/naive_distributed_gcn ogbn-arxiv

metis_dist:
	nvcc --default-stream per-thread -I ../../src/ metis_distr_gcn.cpp ../../src/util/dist_tensor.cu -l cublas  \
	../../src/gnn/dist_sage.cu  ../../src/gnn/sage.cu	\
	../../src/util/tensor.cu  ../../src/dataset.cpp \
	../../src/util/timer.cpp \
	../../src/util/gpu.cu -lnvToolsExt -o ../../build/distributed_gcn
	# ../../build/distributed_gcn ogbn-arxiv metis
