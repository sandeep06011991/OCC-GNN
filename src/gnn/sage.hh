#pragma once
#include <tensor.hh>

class SageAggr{

  Tensor<float> * out_feat  = nullptr;
  Tensor<float> * out_grad = nullptr;

  // offsets refer to out nodes
  Tensor<int> offsets;
  // indices refer to in nodes
  Tensor<int> indices;

  // nodes in are aggregated based on indices and updated in out for forward pass.
  int num_nodes_in;
  int num_nodes_out;

  int fsize;

public:
  SageAggr(int fsize){
      this->fsize = fsize;
  }

  Tensor<float>& forward(Tensor<int>& ind_ptr, Tensor<int>& indices,
          Tensor<float>& in, int num_nodes_out, int num_nodes_in);

  Tensor<float>& backward(Tensor<float> &doutFeat);

};
