#pragma once
#include<tensor.hh>

class DistributedTensor{

private:
  int gpu_ids[4];
  int no_gpus;
  Tensor<float>* local_tensor[4];
  int * ordering;
  Shape s;
  // reordering_ids
  int local_sizes[4];
  int reordering_map*[4];

  void partition(int *ordering){
    int sizes[4] ={0,0,0,0};
    for(int i=0;i<s.dim1;i++){
      sizes[ordering[i]]++;
    }
    for(int i=0;i<no_gpu;i++){
      float *t = malloc(sizeof(float)*sizes[i]*s.dim2);
      
    }
  }

public:
  DistributedTensor(int gpu_ids[], int gpus, int dim1, int dim2, float *data){
    this->no_gpus = gpus;
    for(int i=0;i<gpus;i++){
      this->gpu_ids[i] = gpu_ids[i];
    }
    s(dim1,dim2);
    ordering = malloc(sizeof(int) * dim1 );
    for(int i=0;i<dim1;i++){
      ordering[i] = i% no_gpus;
    }

  }


};
