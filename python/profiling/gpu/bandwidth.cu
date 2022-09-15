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
// Check that I can do a p2p copy
float p2p_copy (size_t size)
{
  //size = sizeof(int);
  cudaSetDevice (0);
  for(size_t i=1;i< 1024 * 4;i = i * 4){
      cudaEvent_t begin, end;
	std::cout << "size in MB" << i <<"\n";

      cudaEventCreate (&begin);
      cudaEventCreate (&end);
	void * host;
      //void *host = malloc(i * 1024 * 1024);
      cudaMallocHost(&host, i * 1024 * 1024);
      void *device;
      gpuErrchk(cudaMalloc(&device,i * 1024 * 1024));
      for(int j=0;j<4;j++){
      cudaEventRecord (begin);
      cudaMemcpy(device, host, i * 1024 * 1024,cudaMemcpyHostToDevice);
      cudaEventRecord (end);
      cudaEventSynchronize (end);
      float elapsed;
      cudaEventElapsedTime (&elapsed, begin, end);
      elapsed /= 1000;
      std::cout << "Data movement" << i << ": " <<  elapsed << " : " << (i * (1.0)/1024)/elapsed <<" GBps\n";
      }
      cudaFree(device);
      cudaFree(host);
      cudaEventDestroy (end);
      cudaEventDestroy (begin);
  }
}

int main(){
  p2p_copy(1024 * 1024 * 1024);
}
