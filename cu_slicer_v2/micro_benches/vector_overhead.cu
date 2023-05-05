#include <thrust/device_vector.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(){
	cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
std::cout <<"sms:"<< deviceProp.multiProcessorCount <<"\n";
std::cout << "mts:" << deviceProp.maxThreadsPerMultiProcessor <<"\n";
cudaSetDevice(0);
	thrust::device_vector<long> data;
	float *t;
	cudaEvent_t event1;
    	cudaEvent_t event2;
	gpuErrchk(cudaEventCreate(&event1));
	gpuErrchk(cudaEventCreate(&event2));
	float milliseconds;
	std::cout << "Rand" << sizeof(curandState) <<"\n";
	return 0;
	for(int i=0;i < 10; i++){
		for(int j=1; j <100000; j = j * 10){
			//gpuErrchk(cudaEventRecord(event1));
			data.resize(j);
			//gpuErrchk(cudaEventRecord(event2));
			//gpuErrchk(cudaEventSynchronize(event2));
			//cudaEventElapsedTime(&milliseconds, event1,event2);
			std::cout << i <<":"<< j <<":" << milliseconds <<"\n";
			
		}
	}
	cudaDeviceSynchronize();

}
