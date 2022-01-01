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
  for(int i=1024;i<1024*1024*1024;i = i * 16){
      cudaEvent_t begin, end;
      cudaEventCreate (&begin);
      cudaEventCreate (&end);
      void *host = malloc(i);
      void *device;
      gpuErrchk(cudaMalloc(&device,i));
      cudaEventRecord (begin);
      cudaMemcpy(device, host, i,cudaMemcpyHostToDevice);
      cudaEventRecord (end);
      cudaEventSynchronize (end);
      float elapsed;
      cudaEventElapsedTime (&elapsed, begin, end);
      elapsed /= 1000;
      cudaFree(device);
      free(host);
      std::cout << "Data movement" << i << " " << elapsed <<"\n";
      cudaEventDestroy (end);
      cudaEventDestroy (begin);
  }
}

int main(){
  p2p_copy(1024 * 1024 * 1024);
}
