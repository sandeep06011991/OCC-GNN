#pragma once
#include "graph/sliced_sample.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include <torch/extension.h>
using namespace std;
namespace py = pybind11;


class PyBipartite{

public:

  torch::Tensor out_degree_local;

  int num_in_nodes_local = 0;
  int num_in_nodes_pulled = 0;
  int num_out_local = 0;
  int num_out_remote = 0;

  torch::Tensor indptr_L;
  torch::Tensor indices_L;
  torch::Tensor indptr_R;
  torch::Tensor indices_R;


  std::vector<torch::Tensor> from_ids;
  std::vector<int> to_offsets;

  vector<torch::Tensor> push_to_ids;
  std::vector<int> pull_from_offsets;

  int self_ids_offset = 0;

  int gpu_id;

  PyBipartite(BiPartite *bp, int local_gpu_id, int num_gpus);

  ~PyBipartite();
};

class PySample{
public:
  std::vector<std::vector<PyBipartite *> *> layers;

  // Pass missing_node_ids
  std::vector<torch::Tensor> cache_hit_from;
  std::vector<torch::Tensor> cache_hit_to;
  std::vector<torch::Tensor>cache_miss_from;
  std::vector<torch::Tensor> cache_miss_to;
  std::vector<torch::Tensor> out_nodes;
  int num_layers = 0;

  std::vector<int> debug_vals;



  PySample(PartitionedSample &s, int current_gpu, int num_gpus);

  ~PySample();
};


inline torch::Tensor getTensor(thrust::device_vector<long> &v, c10::TensorOptions opts){
    if(v.size() == 0){
      return torch::empty(v.size());
    }else{
      return torch::from_blob((long *)thrust::raw_pointer_cast(v.data()), {(long)v.size()}, opts).clone();

    }

}
