bce:
	python3 bce_test.py
	nvcc  -I ../../src/ ../../src/util/tensor.cu ../../src/layers/loss.cu -lcublas \
	bce_test.cc -o ../../build/test
	../../build/test

make: bce
