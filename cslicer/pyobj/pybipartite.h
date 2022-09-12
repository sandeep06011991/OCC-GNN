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
  torch::Tensor in_nodes;
  torch::Tensor expand_indptr;
  torch::Tensor indptr;
  torch::Tensor indegree;
  torch::Tensor out_nodes;
  torch::Tensor owned_out_nodes;

  int num_in_nodes;
  int num_out_nodes;

  // Filled afer reordering
  torch::Tensor indices;

  // Easy fill
  std::vector<torch::Tensor> from_ids;
  std::vector<torch::Tensor> to_ids;

  torch::Tensor self_ids_in;
  torch::Tensor self_ids_out;

  int gpu_id = -1;

  torch::Tensor missing_node_ids;

  PyBipartite(BiPartite *bp);

  ~PyBipartite();
};

class PySample{
public:
  std::vector<std::vector<PyBipartite *> *> layers;

  // Pass missing_node_ids
  std::vector<torch::Tensor> missing_node_ids;

  long in_nodes = 0;
  long out_nodes = 0;

  PySample(PartitionedSample &s);

  ~PySample();
};
