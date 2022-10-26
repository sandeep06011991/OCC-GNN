# Used for local testing of slicer.
# Add more unit tests here.

sample_test:
	g++ -g -I. tests/sample.cpp graph/dataset.cpp samplers/ns.cpp  \
	util/duplicate.cpp transform/slice.cpp graph/bipartite.cpp \
	test_fronend.cpp
	./a.out
all:
	g++ -I. test_metis.cpp  graph/dataset.cpp samplers/ns.cpp transform/walk.cpp \
	util/duplicate.cpp transform/slice.cpp graph/bipartite.cpp


slice_test:
	g++ -I. tests/slice_test1.cpp transform/slice.cpp util/duplicate.cpp\
		graph/bipartite.cpp
	./a.out

frontend:
	g++ test_fronend.cpp graph/dataset.cpp samplers/ns.cpp util/duplicate.cpp \
	transform/slice.cpp graph/bipartite.cpp -I.
	./a.out



conqueue:
		g++ -std=c++11 -pthread -O3 -std=c99 -Wall -Wextra \
			-I . util/conqueue.cpp -o ../build/conqueue
		../build/conqueue
