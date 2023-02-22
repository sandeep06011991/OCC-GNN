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

__global__ void compute_heavy(float *a){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float x = a[j];
	for(int i =0; i < 100; i++){
		for(int jj = 0; jj < 10000; jj ++){
		x = x * 1.200;
		}	
	}
	a[j] = x;
}

int main(){
	cudaSetDevice(0);
	int blocks = 984;
	int threads = 256;
	float *a;
	gpuErrchk(cudaMalloc(&a, sizeof(float) * blocks * threads));
	compute_heavy<<<blocks, threads>>>(a);
	gpuErrchk(cudaDeviceSynchronize());
}
