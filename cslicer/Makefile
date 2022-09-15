# Used for local testing of slicer.
# Add more unit tests here.
slice_test:
	g++ -I. tests/slice_test1.cpp transform/slice.cpp util/duplicate.cpp\
		graph/bipartite.cpp
	./a.out

all:
	g++ test_fronend.cpp graph/dataset.cpp samplers/ns.cpp util/duplicate.cpp \
	transform/slice.cpp graph/bipartite.cpp -I.
	./a.out



conqueue:
		g++ -std=c++11 -pthread -O3 -std=c99 -Wall -Wextra \
			-I . util/conqueue.cpp -o ../build/conqueue
		../build/conqueue
