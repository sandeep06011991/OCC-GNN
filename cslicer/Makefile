# Used for local testing of slicer.
# Add more unit tests here.


all:
	g++ -std=c++11 -pthread -O3 -std=c99 -Wall -Wextra \
		driver.cpp dataset.cpp slicer.cpp bipartite.cpp util/duplicate.cpp  \
			 	-I . -o ../build/cslicer
	../build/cslicer


conqueue:
		g++ -std=c++11 -pthread -O3 -std=c99 -Wall -Wextra \
			-I . util/conqueue.cpp -o ../build/conqueue
		../build/conqueue
		
# sampling:
# 	nvcc -I ../src/ ../src/util/tensor.cu ../src/gnn/sage.cu  \
# 	  ../src/dataset.cpp sampling_test.cc ../src/util/timer.cpp \
# 	-lnvToolsExt  -lcublas -o ../build/sampling
# 	# nvcc -I ../src/ ../src/tensor.cu ../src/samplers/full.cpp -lcublas \
# 	# ../src/dataset.cpp gcn_test.cc -o ../build/gcn_test
# 	../build/sampling
