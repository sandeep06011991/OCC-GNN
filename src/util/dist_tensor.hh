#pragma once
#include "util/tensor.hh"
#include <vector>
#include <assert.h>

// Wrapper class of local tensors.
// Instantiate possibilities
// 1. cpu data and rorder map for input feature vectors.
// Local tensor allocation and deallocation responsibility rests wiwth dist tensor
// 2. outputs of local aggr.
// Local tensors are computed externally , allocation and deallocation responsibility
// rests with the creator. This class serves as a view only
class DistTensor{

private:
  // Call after setting s, no_gpus and global_to_gpu
  // set local_to_gpu, global_to_local from global_to_gpu map
  void create_local_mapping();

public:
  Shape s;
  int no_gpus;
  /* Size of s.dim1 as tensors are partitioned by feature.
  *  Each row in global_tensor is mapped to local_id
  *  local ids are on gpu ids.
  */
  std::vector<int> global_to_local;
  std::vector<int> local_to_global[4];
  std::vector<int> global_to_gpu;
  Tensor<float> * local_tensors[4];

  bool isView;

  DistTensor(float *cpu_data, Shape s, int *reorder_map,int no_gpus);

  DistTensor(Shape s, int *reorder_map,int no_gpus);

  void debugTensor(){
    std::cout << "global_to_local\n";
    for(int i=0;i< global_to_local.size();i++){
      std::cout << global_to_local[i] <<" ";
    }
    std::cout << "\n global_to_gpu\n";
    for(int i=0;i< global_to_gpu.size();i++){
        std::cout << global_to_gpu[i] <<" ";
    }
    std::cout << "\n";
    for(int i=0; i<no_gpus; i++){
      std:: cout << "Debugging gpu" << i <<"\n";
      std:: cout << "Local to global gpu" << i <<"\n";
      for(int j=0;j<local_to_global[i].size();j++){
        std::cout << local_to_global[i][j] << " ";
      }
      local_tensors[i]->viewTensor();
    }
  }

  void setLocalTensor(Tensor<float> * t, int gpu_id){
    assert(gpu_id < no_gpus);
    local_tensors[gpu_id] = t;
  }

  void clearTensor(){
    global_to_local.clear();
    global_to_gpu.clear();
    for(int i=0;i<no_gpus;i++){
      local_to_global[i].clear();
    }

    if(!isView){
      for(int i=0;i<no_gpus;i++){
        local_tensors[i]->clearTensor();
      }
    }
  }
};
