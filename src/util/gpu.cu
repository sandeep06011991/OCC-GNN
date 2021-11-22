#include "util/gpu.hh"
#include "nn_exception.hh"
#include <iostream>
void sync_all_gpus(){
  for(int i=0;i<no_gpus;i++){
      cudaSetDevice(i);
      cudaDeviceSynchronize();
  }
}


void enable_peer_communication(){
  for(int i=0;i<no_gpus;i++){
    cudaSetDevice (i);
    for(int j=0;j<no_gpus;j++){
      if(i!=j){
        // std::cout << "cuda enable peer " << i << j <<"\n";
        int can_access_peer_0_1;
        cudaDeviceCanAccessPeer(&can_access_peer_0_1, i, j);
        printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", i, j, can_access_peer_0_1);

        // cudaDeviceEnablePeerAccess (j, i);
      }
    }
    NNException::throwIfDeviceErrorsOccurred("cuda peer comm setup failed\n");
  }
}
