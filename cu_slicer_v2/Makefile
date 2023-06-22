.PHONY: all
# INCLUDE=-I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub \
# 	-I/home/spolisetty/3rdparty/thrust/dependencies/libcubxx

GTEST_INCLUDE=-I/home/spolisetty_umass_edu/googletest/googletest/include
LGTEST=/home/spolisetty_umass_edu/googletest/build/lib/libgtest.a
# nvcc -arch=sm_70 -Xptxas=-v  test1.cu 

INCLUDE= -I/home/ubuntu/3rdparty/thrust/dependencies/cub $(GTEST_INCLUDE)
DEBUG=-DDEBUG  --extended-lambda
FLAGS=-lstdc++  -G  -g  --maxrregcount=64 --compiler-options -Wall 
#-Xcompiler=-fno-gnu-unique -code=sm_52 -arch=compute_35
# -g -G
# EXTRAFLAGS = -01
# everything:
# 	nvcc -c -I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub \
# 	      -I/home/spolisetty/3rdparty/thrust/dependencies/libcubxx	main.cu\
# 	      graph/dataset.cu  samplers/ns.cu  util/duplicate.cu  graph/bipartite.cu\
# 	      tests/gcn_test.cu  transform/gcn_slice.cu

# all: graph/dataset.o build/main.o util/duplicate.o samplers/ns.o \
# 	 	transform/pull_slice.o transform/push_slice.o\
# 		 transform/slice.o graph/bipartite.o tests/gcn_test.o
# 	nvcc -G graph/dataset.o build/main.o  util/duplicate.o \
# 	 	samplers/ns.o transform/pull_slice.o  transform/push_slice.o transform/slice.o\
# 		 	graph/bipartite.o  tests/gcn_test.o -o a.out
# 	./a.out

# all: build/main.o build/graph/dataset.o build/samplers/ns.o
# 	nvcc -G build/main.o build/graph/dataset.o \
# 		-o cuslicer
# Todo add all this to  regex catching
SRC := $(wildcard util/*.cu graph/*.cu)
OBJ := $(SRC:%.cu=build/%.o)

all: build/main.o build/util/cub.o build/util/duplicate.o build/util/device_vector.o \
			build/graph/dataset.o build/samplers/ns.o build/transform/push_slice.o build/transform/slice.o \
				build/tests/gcn_test.o build/graph/bipartite.o build/transform/pull_slice.o\
			build/util/array_utils.o 
	nvcc -G $(FLAGS) $(DEBUG) build/main.o build/util/cub.o build/util/device_vector.o   build/samplers/ns.o \
			build/graph/dataset.o  build/transform/slice.o build/transform/pull_slice.o   build/tests/gcn_test.o \
	build/transform/push_slice.o  build/util/array_utils.o build/util/duplicate.o build/graph/bipartite.o -o build/cuslicer
	./build/cuslicer
traverse: build/traverse_main.o build/util/cub.o build/util/duplicate.o build/util/device_vector.o \
				build/graph/dataset.o build/samplers/ns.o
	nvcc -G build/traverse_main.o build/util/cub.o build/util/device_vector.o   build/samplers/ns.o \
				build/graph/dataset.o  \
				build/util/duplicate.o -o traverse

tests: build/tests/gtest_entry.o  build/tests/cub_tests.o  build/tests/device_vector.o \
			build/util/cub.o build/util/device_vector.o build/tests/array_map.o build/util/duplicate.o \
				build/tests/sampling.o build/tests/slicing.o build/samplers/ns.o build/graph/dataset.o\
				build/transform/push_slice.o build/transform/slice.o \
					build/tests/gcn_test.o build/graph/bipartite.o
	nvcc -G $(FLAGS) build/tests/gtest_entry.o build/tests/cub_tests.o build/samplers/ns.o  \
		build/util/cub.o  build/tests/array_map.o build/util/duplicate.o build/tests/device_vector.o  \
		build/util/device_vector.o build/tests/sampling.o build/graph/dataset.o\
		$(LGTEST)  -o gtests
	# ./gtests --gtest_filter=CUB.exclusive_sum
	# ./gtests --gtest_filter=ARRAYMAP.memory

build/main.o : main.cu
	mkdir -p build/
	nvcc $(DEBUG) $(FLAGS) $(INCLUDE) main.cu -c -o build/main.o


build/traverse_main.o : traverse_main.cu
	mkdir -p build/
	nvcc $(DEBUG) $(INCLUDE) traverse_main.cu -c -o build/traverse_main.o


build/graph/%.o: graph/%.cu
	mkdir -p build/graph
	nvcc  $(DEBUG)  $(FLAGS) -c $(INCLUDE) $< -o $@

util/%.o: util/%.cu
	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

build/samplers/%.o: samplers/%.cu
	mkdir -p build/samplers
	nvcc  $(DEBUG) $(FLAGS) -c $(INCLUDE) $< -o $@

build/transform/%.o: transform/%.cu
	mkdir -p build/transform
	nvcc $(DEBUG) $(FLAGS) -c $(INCLUDE) $< -o $@

build/tests/%.o: tests/cuda/%.cu
	mkdir -p build/tests
	nvcc $(DEBUG)  -c $(INCLUDE) $< -o $@

build/util/%.o: util/%.cu
	mkdir -p build/util
	nvcc $(DEBUG) $(FLAGS) -c $(INCLUDE) $< $(FLAGS) -o $@


clean:
	rm -r build/*
	rm -f util/*.o
	rm -f graph/*.o
	rm -f samplers/*.o
	rm -f transform/*.o
	rm -f main.o
