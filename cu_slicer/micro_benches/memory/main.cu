#include<iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#include<iostream>

__global__
void f(int *a, size_t t){
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	while(id<t){
	a[id] = 0;
	id += (gridDim.x * blockDim.x);
	}
}
int main(){
	cudaSetDevice(0);
	size_t size = 1 * 1024 * 1024 * 1024/sizeof(int);
	int *a, *b;
	//gpuErrchk(cudaMallocManaged(&a, size * sizeof(int)));
	gpuErrchk(cudaHostAlloc(&b, size * sizeof(int), cudaHostAllocMapped));
	gpuErrchk(cudaHostGetDevicePointer(&a, b, 0));
	std::cout <<"Allocatoopn\n";
	f<<<1,1>>>(a,size);
	gpuErrchk(cudaDeviceSynchronize());
	std::cout <<"kernel ok!\n";
	while(true){
	
	}
}
