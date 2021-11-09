bce:
	nvcc  -I ../src/ ../src/tensor.cu ../src/layers/loss.cu -lcublas \
	bce_test.cc -o ../build/test
	../build/test

linear:
	nvcc -I ../src/ ../src/tensor.cu ../src/layers/linear.cu -lcublas \
	../src/layers/relu.cu linear_test.cc -o ../build/linear_test
	../build/linear_test

matrix:
	nvcc -I ../src/ ../src/tensor.cu -lcublas \
	matrix_util_test.cc -o ../build/mat_util
	../build/mat_util

gcn:
	nvcc -I ../src/ ../src/tensor.cu ../src/gnn/sage.cu -lcublas \
	../src/dataset.cpp gcn_test.cc -o ../build/gcn_test
	# nvcc -I ../src/ ../src/tensor.cu ../src/samplers/full.cpp -lcublas \
	# ../src/dataset.cpp gcn_test.cc -o ../build/gcn_test
	../build/gcn_test

sampling:
	nvcc -I ../src/ ../src/tensor.cu ../src/gnn/sage.cu  \
	  ../src/dataset.cpp sampling_test.cc \
	-lcublas -o ../build/sampling
	# nvcc -I ../src/ ../src/tensor.cu ../src/samplers/full.cpp -lcublas \
	# ../src/dataset.cpp gcn_test.cc -o ../build/gcn_test
	../build/sampling

sampled_gcn:
	nvcc -I ../src/ ../src/dataset.cpp ../src/tensor.cu  \
	../src/gnn/sage.cu ../src/layers/*.cu -lcublas  \
	 sampled_gcn.cc -o ../build/forward
	../build/forward

full_graph_gcn:
	nvcc -I ../src/ ../src/dataset.cpp ../src/tensor.cu  \
	../src/gnn/sage.cu ../src/layers/*.cu -lcublas  \
	 	full_graph_gcn.cc -o ../build/full
	../build/full

single:
	nvcc -I ../src/ single_gpu.cc ../src/tensor.cu  \
	-lcublas  -o ../build/full
	../build/full

dist_gcn:
	nvcc -I ../src/ ../src/util/tensor.cu distributed_gcn.cpp  \
	../src/util/dist_tensor.cu  -lcublas -o ../build/distributed_gcn
	../build/distributed_gcn
