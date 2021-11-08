#pragma once

class DistTensor{

  Shape s;

  int no_gpus;
  int gpu_ids[4];

  /* Size of s.dim1 as tensors are partitioned by feature.
  *  Each row in global_tensor is mapped to local_id
  *
  */
  int * reorder_map;

  Tensor<float> * local_tensors[4];

}
