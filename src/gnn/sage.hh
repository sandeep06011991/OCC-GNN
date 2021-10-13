#pragma once
#include <tensor.hh>

class SageAggr{

  Tensor<float> * out_feat;
  Tensor<float> * out_grad;

  Tensor<int> * offsets;
  Tensor<int> * indices;

  int num_nodes_in;
  int num_nodes_out;
  int fsize;

public:
  SageAggr(int fsize){
      this->fsize = fsize;
  }

  Tensor<float>& forwardPass(Tensor<int> * ind_ptr, Tensor<int> * indices,
          Tensor<float>& in, int num_nodes_out, int num_nodes_in);

  Tensor<float>& backwardPass(Tensor<float> &outFeat);

};
