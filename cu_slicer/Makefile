# Used for local testing of slicer.
# Add more unit tests here.

all:
	nvcc  -I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub \
	      -I/home/spolisetty/3rdparty/thrust/dependencies/libcubxx	main.cu\
	      graph/dataset.cu  samplers/ns.cu  util/duplicate.cu transform/slice.cu
	#transform/slice.cpp graph/bipartite.cpp tests/gcn_test.cpp \
	#tests/gat_test.cpp tests/test_pull.cpp \
	#tests/test_redundancy.cpp transform/walk.cpp
	./a.out
