#pragma once
#include <util/tensor.hh>
#include "util/dist_tensor.hh"
#include <vector>
#include "gnn/local_graph.hh"

using namespace std;


void merge(Tensor<float> *src, Tensor<float> *dest, Tensor<int> indices);

// Contains all the logic for gcn aggregation
// when the data is placed as a distributed tensor.
// returns a distributed tensor.
class DistSageAggr{

  // nodes in are aggregated based on indices and updated in out for forward pass.
  int num_nodes_in;
  int num_nodes_out;
  int fsize;
  int no_gpus;

  // [src],[dest]
  LocalComputeGraph local_graph[4][4];

  void populateLocalGraphs(DistTensor &in, vector<int> &indptr,vector<int> &indices);

public:

  DistTensor * out_feat  = nullptr;
  DistTensor * out_grad = nullptr;

  DistSageAggr(int fsize,int no_gpus){
      this->fsize = fsize;
      this->no_gpus = no_gpus;
      for(int i=0;i<no_gpus;i++){
        for(int j=0; j<no_gpus;j++){
          local_graph[i][j].set_src_dest(i,j);
        }
      }
  };

  // void test(vector<int>& a);

  void forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in);

  DistTensor& backward(DistTensor& doutFeat);

  merge(temp[dest][dest],temp[src][dest],this->local_graph[src][dest].local_to_local[id]);

};
