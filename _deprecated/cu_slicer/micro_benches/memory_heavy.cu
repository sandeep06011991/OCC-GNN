#include <iostream>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void compute_heavy(float *a, size_t sz){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float x = a[j];
	
	for(int i =0; i < 100; i++){
		for(int jj = 0; jj < 10000; jj ++){
		x ++ ;
		//j = j + blockDim.x;
		//j = j % sz;
		}	
	}
	a[j] = x;
}

int main(){
	cudaSetDevice(0);
	int blocks = 984;
	int threads = 256;
	float *a;
	size_t sz = 1032 * 1032 * 1032/4;
	gpuErrchk(cudaMalloc(&a, sizeof(float) * sz));
	compute_heavy<<<blocks, threads>>>(a,sz);
	gpuErrchk(cudaDeviceSynchronize());
}
