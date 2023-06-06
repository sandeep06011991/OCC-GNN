#include <iostream>
#include <cassert>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include <curand.h>
#include <curand_kernel.h>

int main(){
	std::cout << "Checl" <<  sizeof(curandState) <<" " << sizeof(int) <<  "\n";
	int * indptr_h;
	int * indptr_d;
	int num_nodes = 1000 * 1000 * 1000;
//gpuErrchk(cudaMalloc(&indptr_d, (num_nodes) * sizeof(long)));
gpuErrchk(cudaHostAlloc(&indptr_h,(num_nodes + 1) * sizeof(long), cudaHostAllocMapped | cudaHostAllocWriteCombined ));
    //file1.read((char *)indptr_h,(this->num_nodes + 1) * sizeof(NDTYPE));
    gpuErrchk(cudaHostGetDevicePointer(&indptr_d, indptr_h, 0));
	while(true){
	} 
}
