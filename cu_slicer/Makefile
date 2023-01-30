.PHONY: all
INCLUDE=-I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub \
	-I/home/spolisetty/3rdparty/thrust/dependencies/libcubxx
DEBUG=-DDEBUG
# everything:
# 	nvcc -c -I. -I/home/spolisetty/3rdparty/thrust -I/home/spolisetty/3rdparty/thrust/dependencies/cub \
# 	      -I/home/spolisetty/3rdparty/thrust/dependencies/libcubxx	main.cu\
# 	      graph/dataset.cu  samplers/ns.cu  util/duplicate.cu  graph/bipartite.cu\
# 	      tests/gcn_test.cu  transform/gcn_slice.cu

all: graph/dataset.o main.o util/duplicate.o samplers/ns.o \
	 	transform/pull_slice.o transform/push_slice.o transform/slice.o graph/bipartite.o
	nvcc -G graph/dataset.o main.o  util/duplicate.o \
	 	samplers/ns.o transform/pull_slice.o  transform/push_slice.o transform/slice.o\
		 	graph/bipartite.o -o a.out
	# ./a.out

main.o: main.cu
	nvcc $(DEBUG) $(INCLUDE) main.cu -c -o main.o

graph/%.o: graph/%.cu
	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

util/%.o: util/%.cu
	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

samplers/%.o: samplers/%.cu
	nvcc  $(DEBUG) -c $(INCLUDE) $< -o $@

transform/%.o: transform/%.cu
	nvcc $(DEBUG) -c $(INCLUDE) $< -o $@

clean:
	rm -f util/*.o
	rm -f graph/*.o
	rm -f samplers/*.o
	rm -f transform/*.o
	rm -f main.o
