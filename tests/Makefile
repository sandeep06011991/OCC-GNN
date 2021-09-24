bce:
	nvcc  -I ../src/ ../src/tensor.cu ../src/layers/loss.cu  \
	bce_test.cc -o ../build/test
	../build/test

linear:
	nvcc -I ../src/ ../src/tensor.cu ../src/layers/linear.cu \
	../src/layers/relu.cu linear_test.cc -o ../build/linear_test
	../build/linear/test
