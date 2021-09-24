all:
	nvcc -I src/ -I src/layers src/dataset.cpp src/driver.cu  src/tensor.cu \
	src/layers/linear.cu src/layers/relu.cu -o build/driver
	./build/driver

# all:
# 		nvcc -I src/ src/driver.cu src/graph.cpp -o build/driver
# 		./build/driver
