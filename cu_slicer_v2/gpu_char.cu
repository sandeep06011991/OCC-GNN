#include <stdio.h>

int main() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  maxBlocksPerMultiProcessor : %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  maxThreadsPerBlock: %d \n", prop.maxThreadsPerBlock);
    printf("  maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("  registersPerMultiProcessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    //printf("  gpuDirectRDMASupported: %s\n", prop.gpuDirectRDMASupported ? "yes" : "no");
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  } 
}  
