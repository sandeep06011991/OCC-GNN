#include<chrono>
#include <iostream>
#include <string>
#include <cstring>

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
 auto start1 = std::chrono::system_clock::now();
     for(int i=0;i<100;i++){
        for(int j=0;j<4;j++){
                cudaSetDevice(j);
                // void * device_array;
                // int num_bytes = 1<<10;
                // gpuErrchk(cudaMalloc((void**)&device_array, num_bytes));
                // cudaFree(device_array);
             cudaDeviceSynchronize();
            }
        }
  auto end1 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - start1;
  std::cout << "DeviceSync: " << elapsed1.count()/100 << "s\n";
  cudaDeviceSynchronize();
  cudaStream_t stream[4];
  for(int j=0;j<4;j++){
    (cudaStreamCreate(&stream[j]));
  }
  auto start2 = std::chrono::system_clock::now();
     for(int j=0;j<4;j++){
       auto start2= std::chrono::system_clock::now();
       for(int i=0;i<100;i++){
         cudaSetDevice(j);
         gpuErrchk(cudaStreamSynchronize(stream[j]));
     }
   }
   auto end2 = std::chrono::system_clock::now();
   std::chrono::duration<double> elapsed2 = end2 - start2;
   std::cout << "Stream sync: " << elapsed2.count()/100 << "s\n";

   int num_bytes = 1024 * 1024 * 1024;
   cudaStream_t stream1;
   cudaStreamCreate(&stream1);
   void * device_array;
   void * host_array = (void *)malloc(num_bytes);
   gpuErrchk(cudaMalloc((void**)&device_array, num_bytes));
   for(int i=1024;i<1024*1024*1024;i=i*8){
     auto start3= std::chrono::system_clock::now();
     gpuErrchk(cudaMemcpyAsync(device_array,host_array,i,cudaMemcpyHostToDevice,stream1));
     cudaStreamSynchronize(stream1);
     auto end3 = std::chrono::system_clock::now();
     std::chrono::duration<double> elapsed3 = end3 - start3;
     std::cout << "data moevemt: "<< i * 1.0/(1024 * 1024) <<"MB "<< elapsed3.count() << "s\n";

   }
}
