.PHONY: all
# INCLUDE=-I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub \
# 	-I/home/spolisetty/3rdparty/thrust/dependencies/libcubxx

INCLUDE= -I/home/spolisetty/3rdparty/thrust/dependencies/cub
DEBUG=-DDEBUG  --extended-lambda
FLAGS=-lstdc++
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
			build/graph/dataset.o build/samplers/ns.o
	nvcc -G build/main.o build/util/cub.o build/util/device_vector.o   build/samplers/ns.o \
			build/graph/dataset.o \
	build/util/duplicate.o -o cuslicer

tests: build/tests/test1_fact.o  build/tests/cub_tests.o  build/tests/device_vector.o \
			build/util/cub.o build/util/device_vector.o build/tests/array_map.o build/util/duplicate.o \
				build/tests/sampling.o build/samplers/ns.o build/graph/dataset.o
	nvcc -G  build/tests/test1_fact.o build/tests/cub_tests.o build/samplers/ns.o  \
		build/util/cub.o  build/tests/array_map.o build/util/duplicate.o build/tests/device_vector.o  \
		build/util/device_vector.o build/tests/sampling.o build/graph/dataset.o\
		-lgtest -o gtests
	# ./gtests --gtest_filter=CUB.exclusive_sum
	./gtests --gtest_filter=ARRAYMAP.memory

build/main.o : main.cu
	mkdir -p build/
	nvcc $(DEBUG) $(INCLUDE) main.cu -c -o build/main.o

build/graph/%.o: graph/%.cu
	mkdir -p build/graph
	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

# util/%.o: util/%.cu
# 	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

build/samplers/%.o: samplers/%.cu
	mkdir -p build/samplers
	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

transform/%.o: transform/%.cu
	nvcc $(DEBUG) -c $(INCLUDE) $< -o $@

build/tests/%.o: tests/%.cu
	mkdir -p build/tests
	nvcc $(DEBUG) -c $(INCLUDE) $< -o $@

build/util/%.o: util/%.cu
	mkdir -p build/util
	nvcc $(DEBUG) -c $(INCLUDE) $< $(FLAGS) -o $@


clean:
	rm -r build/*
	rm -f util/*.o
	rm -f graph/*.o
	rm -f samplers/*.o
	rm -f transform/*.o
	rm -f main.o
