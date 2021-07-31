all:
		nvcc src/driver.cu -o build/driver
		./build/driver
