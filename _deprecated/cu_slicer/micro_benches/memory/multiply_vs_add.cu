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
	//int id = 0;
	int b;
	while(id<t){
	a[id] = 0;
	//id += 1;
	id += (gridDim.x * blockDim.x);
	}
}

int device_properties() {

  int nDevices = 1;

  printf("Number of devices: %d\n", nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Clock Rate (MHz): %d\n", prop.clockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Max Thread per Block (GB/s): %d\n", prop.maxThreadsPerBlock);
    printf("  Multi Processor Count : %d\n",  prop.multiProcessorCount);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }
}

int main(){
	cudaSetDevice(0);
	device_properties();
	size_t size = 100 * 1024 * 1024 /sizeof(int);
	int cache_lines = size * sizeof(int) / 128;
	int *a, *b;
	gpuErrchk(cudaMalloc(&a, size * sizeof(int)));
	//gpuErrchk(cudaHostAlloc(&b, size * sizeof(int), cudaHostAllocMapped));
	//gpuErrchk(cudaHostGetDevicePointer(&a, b, 0));
	std::cout <<"Allocatoopn\n";
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	for(int i=0;i<10;i++){
		cudaEventRecord(start);
		f<<<128,32>>>(a,size);
		cudaEventRecord(stop);
		gpuErrchk(cudaEventSynchronize(stop));
		gpuErrchk(cudaDeviceSynchronize());
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "Milli Seconds:" << milliseconds <<"\n";
	}
	std::cout <<"kernel ok!\n";
}
