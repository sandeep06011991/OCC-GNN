#pragma once
#include <util/tensor.hh>
#include "util/dist_tensor.hh"
#include <vector>
#include "gnn/local_graph.hh"

using namespace std;

// Src matrix is merged with desitnation matrix using idices offsets.
// sizeof(indices.dim1) == src.dim1;
void merge(Tensor<float> *src, Tensor<float> *dest, Tensor<int> *indices);

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
  bool israndomPartitioning;
  bool isExternalPartitioning;
  void populateLocalGraphs(DistTensor &in, vector<int> &indptr,vector<int> &indices);

public:

  DistTensor * out_feat  = nullptr;
  DistTensor * out_grad = nullptr;

  cudaEvent_t start;
  cudaEvent_t stop;
  
  DistSageAggr(int fsize,int no_gpus, bool isRandom, bool isExternal){
      this->fsize = fsize;
      this->no_gpus = no_gpus;
      for(int i=0;i<no_gpus;i++){
        for(int j=0; j<no_gpus;j++){
          local_graph[i][j].set_src_dest(i,j,fsize);
        }
      }
      this->israndomPartitioning = isRandom;
      this->isExternalPartitioning = isExternal;

      // float dur[4];
      for(int i=0;i<1;i++){
        cudaSetDevice(i);
        auto error = cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto  error_1 = cudaEventRecord(start);
        cudaEventSynchronize(start);
      }
  };

  // void test(vector<int>& a);

  void forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in,int *ext_map);

  DistTensor& backward(DistTensor& doutFeat);

};
