all:
		nvcc -I src/ src/driver.cu src/graph.cpp -o build/driver
		./build/driver
