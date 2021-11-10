#pragma once
#include <util/tensor.hh>
#include "util/dist_tensor.hh"
#include <vector>
using namespace std;


struct{
  Tensor<int> *indptr = nullptr;
  Tensor<int> *indices = nullptr;
} csr

// Contains all the logic for gcn aggregation
// when the data is placed as a distributed tensor.
// returns a distributed tensor.
class DistSageAggr{

  // nodes in are aggregated based on indices and updated in out for forward pass.
  int num_nodes_in;
  int num_nodes_out;
  int fsize;

  csr local_csrs[4][4];

  Sage  

public:

  DistTensor * out_feat  = nullptr;
  DistTensor * out_grad = nullptr;

  DistSageAggr(int fsize){
      this->fsize = fsize;
  };

  // void test(vector<int>& a);

  void forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in);

  DistTensor& backward(DistTensor& doutFeat);

};
