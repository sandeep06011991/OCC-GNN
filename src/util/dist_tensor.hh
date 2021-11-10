#pragma once
#include "util/tensor.hh"

class DistTensor{

  Shape s;
  int no_gpus;
  int gpu_ids[4];
  /* Size of s.dim1 as tensors are partitioned by feature.
  *  Each row in global_tensor is mapped to local_id
  */
  int * global_to_local[4];
  int * local_to_global[4];
  Tensor<float> * local_tensors[4];

public:
  DistTensor(float *cpu_data, Shape s, int *reorder_map);

  void debugTensor();

};
