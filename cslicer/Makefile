

all:
	nvcc driver.cpp dataset.cpp -o ../build/cslicer
	../build/cslicer

# sampling:
# 	nvcc -I ../src/ ../src/util/tensor.cu ../src/gnn/sage.cu  \
# 	  ../src/dataset.cpp sampling_test.cc ../src/util/timer.cpp \
# 	-lnvToolsExt  -lcublas -o ../build/sampling
# 	# nvcc -I ../src/ ../src/tensor.cu ../src/samplers/full.cpp -lcublas \
# 	# ../src/dataset.cpp gcn_test.cc -o ../build/gcn_test
# 	../build/sampling
