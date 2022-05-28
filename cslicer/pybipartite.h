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
  std::vector<long> in_nodes_v;
  torch::Tensor in_nodes;
  
  std::vector<long> expand_indptr_v;
  torch::Tensor expand_indptr;  
  std::vector<long> indptr_v;
  torch::Tensor indptr;
  std::vector<long> out_nodes_v;
  torch::Tensor out_nodes;

  std::vector<long> owned_out_nodes_v;
  torch::Tensor owned_out_nodes;
  int num_in_nodes;
  int num_out_nodes;

  // Filled afer reordering
  std::vector<long> indices_v;
  torch::Tensor indices;

  // Easy fill
  std::vector<std::vector<long>> from_ids_v;
  std::vector<torch::Tensor> from_ids;
  std::vector<std::vector<long>> to_ids_v;
  std::vector<torch::Tensor> to_ids;	 
  
  std::vector<long> self_ids_in_v;
  torch::Tensor self_ids_in;
  std::vector<long> self_ids_out_v;
  torch::Tensor self_ids_out;

  int gpu_id = -1;


  PyBipartite(BiPartite *bp);

  ~PyBipartite();
};

class PySample{
public:
  std::vector<std::vector<PyBipartite *> *> layers;

  long in_nodes = 0;
  long out_nodes = 0;

  PySample(Sample *s);

  ~PySample();
};

template class ConQueue<PySample *>;

// void testconqueue(){
//   ConQueue<PySample *> * obj = new ConQueue<PySample *>(10);
// }
