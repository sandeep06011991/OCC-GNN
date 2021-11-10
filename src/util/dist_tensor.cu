#include "util/dist_tensor.hh"
#include "nn_exception.hh"
#include <cstring>
#include <iostream>

// slice data continuosly into local maps and place it. 
DistTensor::DistTensor(float *cpu_data, Shape s, int *reorder_map){

  this->s = s;

  int local_sizes[4];
  memset(local_sizes, 0, sizeof(int) * 4);
  for(int i=0;i<s.dim1;i++){
    assert(reorder_map[i] < 4);
    local_sizes[reorder_map[i]]++;
  }

  float * local_slice[4];
  int * local_to_global[4];
  memset(local_slice, 0, sizeof(float *) * 4);
  memset(local_to_global,0, sizeof(int *) * 4);
  for(int i=0;i<4;i++){
      local_slice[i] = (float *)malloc(sizeof(float) * s.dim2 * local_sizes[i]);
      local_to_global[i] = (int *)malloc(sizeof(int) * s.dim2 * local_sizes[i]);
  }

  int indptr[4];
  memset(indptr, 0, sizeof(int) * 4);
  for(int i=0;i<s.dim1;i++){
    int pos = reorder_map[i];
    memcpy(&local_slice[pos][indptr[pos]], &cpu_data[i*s.dim2], sizeof(float) * s.dim2);
    indptr[pos] = indptr[pos] + s.dim2;
  }

  for(int i=0; i<4; i++){
    Shape local_s(local_sizes[i],s.dim2);
    local_tensors[i] = new Tensor<float>(local_slice[i], local_s, i);
  }

  NNException::throwIfDeviceErrorsOccurred("dist tensor initialization failed\n");

}

void DistTensor::debugTensor(){
  for(int i=0;i<4;i++){
    local_tensors[i]->debugTensor();
  }
}
