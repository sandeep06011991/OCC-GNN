#pragma once
#include <tensor.hh>
#include <util/dist_tensor.hh>

// Contains all the logic for gcn aggregation
// when the data is placed as a distributed tensor.
class DistSageAggr{

  // nodes in are aggregated based on indices and updated in out for forward pass.
  int num_nodes_in;
  int num_nodes_out;
  int fsize;

public:

  DistTensor * out_feat  = nullptr;
  DistTensor * out_grad = nullptr;

  DistSageAggr(int fsize){
      this->fsize = fsize;
  }

  DistTensor<float>& forward(Tensor<int>& ind_ptr, Tensor<int>& indices,
          DTensor<float>& in, int num_nodes_out, int num_nodes_in);

  DistTensor& backward(DistTensor &doutFeat);

};
