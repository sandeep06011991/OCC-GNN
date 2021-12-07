#include "util/dist_tensor.hh"
#include "nn_exception.hh"
#include <cstring>
#include <iostream>
#include "util/gpu.hh"
#include "util/timer.h"

void DistTensor::create_local_mapping(){
  this->global_to_local.resize(s.dim1);
  // size of local tensor on each gpu.
  int local_sizes[4];
  memset(local_sizes, 0, sizeof(int) * 4);
  for(int i=0;i<s.dim1;i++){
    assert(this->global_to_gpu[i] < this->no_gpus);
    local_sizes[this->global_to_gpu[i]]++;
  }

  for(int i=0;i<4;i++){
      local_to_global[i].resize(local_sizes[i]);
  }

  int indptr[4];
  memset(indptr, 0, sizeof(int) * 4);
  for(int i=0;i<s.dim1;i++){
    int pos = this->global_to_gpu[i];
    this->global_to_local[i] = indptr[pos];
    this->local_to_global[pos][indptr[pos]] = i;
    indptr[pos] = indptr[pos] + 1;
  }
}

// slice data continuosly into local maps and place it.
// Used to create H_0. The input feature vectors.
DistTensor::DistTensor(float *cpu_data, Shape s, int *reorder_map, int no_gpus){
  isView = false;
  this->s = s;
  this->no_gpus = no_gpus;
  this->global_to_gpu.resize(s.dim1);
  copy(reorder_map, reorder_map + s.dim1, this->global_to_gpu.begin());
  create_local_mapping();
  float * local_slice[4];
  for(int i=0;i<no_gpus;i++){
      local_slice[i] = (float *)malloc(sizeof(float) * s.dim2 * this->local_to_global[i].size());
  }

  int indptr[4];
  memset(indptr, 0, sizeof(int) * 4);
  for(int i=0;i<no_gpus;i++){
    for(int j=0;j<local_to_global[i].size();j++){
      memcpy(&local_slice[i][j * s.dim2], &cpu_data[local_to_global[i][j]*s.dim2], sizeof(float) * s.dim2);
    }
  }

  // start_timer(MOVEMENT_COST);
  for(int i=0; i<no_gpus; i++){
    Shape local_s(this->local_to_global[i].size(),s.dim2);
    this->local_tensors[i] = new Tensor<float>(local_slice[i], local_s, i);
  }
  sync_all_gpus();
  // stop_timer(MOVEMENT_COST);

  // Free temporary data structures.
  for(int i=0;i<no_gpus;i++){
    free(local_slice[i]);
  }
  NNException::throwIfDeviceErrorsOccurred("dist tensor initialization failed\n");
}

DistTensor::DistTensor(Shape s, int *reorder_map,int no_gpus){
  isView = true;
  this->s = s;
  this->no_gpus = no_gpus;
  this->global_to_gpu.resize(s.dim1);
  copy(reorder_map, reorder_map + s.dim1, this->global_to_gpu.begin());
  create_local_mapping();
  for(int i=0;i<no_gpus;i++){
    this->local_tensors[i] = nullptr;
  }
}

void  DistTensor::viewTensor(){
  float * result = (float *)malloc(sizeof(float) * s.dim1 * s.dim2);
  for(int i=0;i<4;i++){
    Shape local_s = this->local_tensors[i]->s;
    float * mem = (float *)malloc(sizeof(float)*local_s.dim1 * local_s.dim2);
    this->local_tensors[i]->copyTensorToCPUMemory(mem);
    for(int j=0;j<local_s.dim1;j++){
      int actual = this->local_to_global[i][j];
      memcpy(&result[actual * s.dim2], &mem[j * s.dim2] , sizeof(float) * s.dim2);
    }
    free(mem);
  }
  int ii = min(3,s.dim1);
  int jj = min(3,s.dim2);
  for(int i=0;i < ii;i++){
    for(int j=0;j< jj ;j++){
      std::cout << result[i*s.dim2 + j ] <<" ";
    }
    std::cout << "\n";
  }
  int sum = 0;
  for(int i=0;i<s.dim1;i++){
    sum = sum + result[i*s.dim2];
  }
  std::cout << "Total is " << sum <<"\n";
  free(result);
  // return result;
}
