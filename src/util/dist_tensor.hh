#pragma once
#include "util/tensor.hh"
#include <vector>
#include <assert.h>
class DistTensor{
public:
  Shape s;
  int no_gpus;
  int gpu_ids[4];
  /* Size of s.dim1 as tensors are partitioned by feature.
  *  Each row in global_tensor is mapped to local_id
  */
  std::vector<int> global_to_local;
  std::vector<int> local_to_global[4];
  std::vector<int> global_to_gpu;
  Tensor<float> * local_tensors[4];


  DistTensor(float *cpu_data, Shape s, int *reorder_map);

  DistTensor(Shape s, int *reorder_map){
    assert(false);
  }

  void debugTensor();

  void clearTensor(){
    for(int i=0;i<no_gpus;i++){
      // Clear everything
    }
  }
};
