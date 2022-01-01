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
            cudaDeviceSynchronize();

                }
        }
  auto end1 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - start1;
  std::cout << "Elapsed time2: " << elapsed1.count() << "s\n";
  cudaStream_t stream[4];
  for(int j=0;j<4;j++){
    (cudaStreamCreate(&stream[j]));
  }
  auto start2= std::chrono::system_clock::now();
  for(int i=0;i<100;i++){
     for(int j=0;j<4;j++){
         cudaSetDevice(j);
         gpuErrchk(cudaStreamSynchronize(stream[j]));
     }
   }
   auto end2 = std::chrono::system_clock::now();
   std::chrono::duration<double> elapsed2 = end2 - start2;
   std::cout << "Stream time2: " << elapsed2.count() << "s\n";
}
