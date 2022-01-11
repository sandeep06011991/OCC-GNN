#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "nvToolsExt.h"
#include <chrono>
int main(void)
{
  // generate 32M random numbers serially
  thrust::host_vector<int> h_vec(32 << 10);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  // transfer data to the device
  thrust::device_vector<int> d_vec;
  d_vec.resize(32<<10);
  nvtxRangePushA("thrust");
  auto start2= std::chrono::system_clock::now();

  for(int i=0;i<100;i++){
    d_vec.clear();
    // std::cout << "size " << d_vec.size()  << " " << d_vec.capacity() <<"\n";
    d_vec = h_vec;
  }
  auto end2 = std::chrono::system_clock::now();
 std::chrono::duration<double> elapsed2 = end2 - start2;
 std::cout << "thrust vector: " << elapsed2.count() << "s\n";
 nvtxRangePop();

 nvtxRangePushA("malloc");
 int *d;

 auto start3= std::chrono::system_clock::now();
 for(int i=0;i<100;i++){
   cudaSetDevice(0);
   cudaMalloc((void **)&d,32 << 10);
   cudaMemcpy(d,h_vec.data(),(32<<10) * sizeof(int),cudaMemcpyHostToDevice);
   cudaFree(d);
   // cudaDeviceSynchronize();
}
 auto end3 = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed3 = end3 - start3;
nvtxRangePop();
std::cout << "naive time: " << elapsed3.count() << "s\n";


  // thrust::device_vector<int> d_vec; = h_vec;

  // sort data on the device (846M keys per second on GeForce GTX 480)
  // thrust::sort(d_vec.begin(), d_vec.end());

  // transfer data back to host
  // thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  std::cout << "hello world\n";
  return 0;
}
