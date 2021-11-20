#pragma once

int no_gpus = 2;

void sync_all_gpus(){
  for(int i=0;i<no_gpus;i++){
      cudaSetDevice(i);
      cudaDeviceSynchronize();
  }
}
