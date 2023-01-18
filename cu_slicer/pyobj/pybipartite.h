#pragma once
#include "graph/sliced_sample.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "util/conqueue.h"
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

  PyBipartite(BiPartite *bp);

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



  PySample(PartitionedSample &s);

  ~PySample();
};
