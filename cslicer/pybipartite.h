#pragma once
#include "sample.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
namespace py = pybind11;

class PyBipartite{
public:
  py::list in_nodes;
  py::list indptr;
  py::list out_nodes;
  py::list owned_out_nodes;

  // Filled afer reordering
  py::list indices;
  // Easy fill
  py::list from_ids;
  py::list to_ids;

  py::list self_ids_in;
  py::list self_ids_out;

  int gpu_id = -1;


  PyBipartite(BiPartite *bp){
    this->gpu_id = bp->gpu_id;
    in_nodes = py::cast(bp->in_nodes);
    indptr = py::cast(bp->indptr);
    out_nodes = py::cast(bp->out_nodes);
    owned_out_nodes = py::cast(bp->owned_out_nodes);
    indices = py::cast(bp->indices);
    for(int i=0;i<4;i++){
      from_ids.append(py::cast(bp->from_ids[i]));
    }
    for(int i=0;i<4;i++){
      to_ids.append(py::cast(bp->to_ids[i]));
    }
    self_ids_in  = py::cast(bp->self_ids_in);
    self_ids_out = py::cast(bp->self_ids_out);
  }
};

class PySample{
public:
  py::list layers;

  int in_nodes = 0;
  int out_nodes = 0;

  PySample(Sample *s){
    for(int i=0;i<3;i++){
        py::list all_bipartites;
        for(int j=0; j<4; j++){
          all_bipartites.append(new PyBipartite(s->layers[i].bipartite[j]));
          if(i==0){
            in_nodes = in_nodes + s->layers[i].bipartite[j]->in_nodes;
          }
          if(j==2){
            out_nodes = out_nodes + s->layers[i].bipartite[j]->out_nodes;
          }
        }
      layers.append(all_bipartites);
    }
  }

  ~PySample(){
    std::cout << "Code to check that destructor is being called\n";
  }
};
