bce:
	nvcc  -I ../src/ ../src/tensor.cu ../src/layers/loss.cu  \
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
