#pragma once
#include "sample.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "util/conqueue.h"
#include <torch/extension.h>
using namespace std;
namespace py = pybind11;

class PyBipartite{
public:
  std::vector<long> data;
  torch::Tensor data_tensor;

  std::vector<long> in_nodes_v;
  torch::Tensor in_nodes;
  int in_nodes_start;
  int in_nodes_end;


  std::vector<long> expand_indptr_v;
  torch::Tensor expand_indptr;
  int expand_indptr_start;
  int expand_indptr_end;

  std::vector<long> indptr_v;
  torch::Tensor indptr;
  int indptr_start;
  int indptr_end;

  std::vector<int> indegree_v;
  torch::Tensor indegree;
  int indegree_start;
  int indegree_end;

  std::vector<long> out_nodes_v;
  torch::Tensor out_nodes;
  int out_nodes_start;
  int out_nodes_end;


  std::vector<long> owned_out_nodes_v;
  torch::Tensor owned_out_nodes;
  int owned_out_nodes_start;
  int owned_out_nodes_end;

  int num_in_nodes;
  int num_out_nodes;

  // Filled afer reordering
  std::vector<long> indices_v;
  torch::Tensor indices;
  int indices_start;
  int indices_end;


  // Easy fill
  std::vector<std::vector<long>> from_ids_v;
  std::vector<long> from_ids_start;
  std::vector<long> from_ids_end;
  std::vector<torch::Tensor> from_ids;
  std::vector<std::vector<long>> to_ids_v;
  std::vector<torch::Tensor> to_ids;
  std::vector<long> to_ids_start;
  std::vector<long> to_ids_end;

  std::vector<long> self_ids_in_v;
  torch::Tensor self_ids_in;
  int self_ids_in_start;
  int self_ids_in_end;
  std::vector<long> self_ids_out_v;
  torch::Tensor self_ids_out;
  int self_ids_out_start;
  int self_ids_out_end;

  int gpu_id = -1;

  std::vector<long> missing_node_ids;

  PyBipartite(BiPartite *bp);

  ~PyBipartite();
};

class PySample{
public:
  std::vector<std::vector<PyBipartite *> *> layers;

  // Pass missing_node_ids
  std::vector<vector<long>> missing_node_ids;

  long in_nodes = 0;
  long out_nodes = 0;

  PySample(Sample *s);

  ~PySample();
};

template class ConQueue<PySample *>;

// void testconqueue(){
//   ConQueue<PySample *> * obj = new ConQueue<PySample *>(10);
// }
