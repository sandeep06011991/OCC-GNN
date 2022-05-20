#pragma once
#include "sample.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "util/conqueue.h"

namespace py = pybind11;

class PyBipartite{
public:
  std::vector<long> in_nodes;
  std::vector<long> indptr;
  std::vector<long> out_nodes;
  std::vector<long> owned_out_nodes;
  int num_in_nodes;
  int num_out_nodes;

  // Filled afer reordering
  std::vector<long> indices;
  // Easy fill
  std::vector<std::vector<long>> from_ids;
  std::vector<std::vector<long>> to_ids;

  std::vector<long> self_ids_in;
  std::vector<long> self_ids_out;

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
