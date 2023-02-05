all:
		nsys profile --stats=true -t cuda ./a.out
		nvcc -I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub main.cu
		sudo /usr/local/cuda-11.1/bin/ncu ./a.out

