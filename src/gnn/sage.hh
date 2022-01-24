#pragma once
#include "util/tensor.hh"

class SageAggr{

  // offsets refer to out nodes
  Tensor<int> offsets;
  // indices refer to in nodes
  Tensor<int> indices;

  // nodes in are aggregated based on indices and updated in out for forward pass.
  int num_nodes_in;
  int num_nodes_out;

  int fsize;
  int device_id;
public:

  Tensor<float> * out_feat  = nullptr;
  Tensor<float> * out_grad = nullptr;

   cudaEvent_t start;
   cudaEvent_t stop;

  SageAggr(int fsize,int device_id){
      this->fsize = fsize;
      this->device_id = device_id;
	  cudaSetDevice(device_id);
    auto error = cudaEventCreate(&this->start);
  	cudaEventCreate(&this->stop);
  }

  Tensor<float> * forward(Tensor<int>& ind_ptr, Tensor<int>& indices,
          Tensor<float>& in, int num_nodes_out, int num_nodes_in);

  Tensor<float>& backward(Tensor<float> &doutFeat);

};
