#include "pybipartite.h"
// #include "sample.h"
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <iostream>
// #include "util/conqueue.h"
#include <pybind11/stl.h>
namespace py = pybind11;

PyBipartite::PyBipartite(BiPartite *bp){
    // std::cout << "attempt bipartite\n";
    // std::cout << bp->gpu_id <<"\n";
    // std::cout << bp->in_nodes.size() <<"\n";
    this->gpu_id = bp->gpu_id;
    in_nodes = bp->in_nodes;
    indptr = bp->indptr;
    num_in_nodes = bp->num_in_nodes;
    num_out_nodes = bp->num_out_nodes;
    std::cout << "Cbipartite" << bp->indices.size() << "\n";
    out_nodes = bp->out_nodes;
    owned_out_nodes = bp->owned_out_nodes;
    indices = bp->indices;
    for(int i=0;i<4;i++){
      from_ids.push_back(bp->from_ids[i]);
      to_ids.push_back(bp->to_ids[i]);
    }
    self_ids_in = bp->self_ids_in;
    self_ids_out = bp->self_ids_out;
    // indptr = py::cast(bp->indptr);
    // std::cout << "bipartite succeess\n";
    // out_nodes = py::cast(bp->out_nodes);
    // owned_out_nodes = py::cast(bp->owned_out_nodes);
    // indices = py::cast(bp->indices);
    // std::cout << "bipartite succeess\n";
    // for(int i=0;i<4;i++){
    //   from_ids.append(py::cast(bp->from_ids[i]));
    // }
    // std::cout << "bipartite succeess\n";
    // for(int i=0;i<4;i++){
    //   to_ids.append(py::cast(bp->to_ids[i]));
    // }
    // self_ids_in  = py::cast(bp->self_ids_in);
    // self_ids_out = py::cast(bp->self_ids_out);
    // std::cout << "bipartite succeess\n";

}

PyBipartite::~PyBipartite(){
  std::cout << "destroy bipartite\n";
}

PySample::PySample(Sample *s){
    for(int i=0;i<3;i++){
        auto all_bipartites = new std::vector<PyBipartite *>();
        // py::list all_bipartites;
        for(int j=0; j<4; j++){
          // std::cout <<"Working on" << j <<"\n";
          // std::cout << s->layers[i].bipartite[j] <<"\n";
          all_bipartites->push_back(new PyBipartite(s->layers[i].bipartite[j]));
          if(i==0){
            in_nodes = in_nodes + s->layers[i].bipartite[j]->in_nodes.size();
          }
          if(j==2){
            out_nodes = out_nodes + s->layers[i].bipartite[j]->out_nodes.size();
          }
        }
        layers.push_back(all_bipartites);
    }
  }

PySample::~PySample(){
    for(auto l:layers){
      for(auto ll:*l){
        delete ll;
      }
      delete l;
    }
    std::cout << "Code to check that destructor is being called\n";
}

void testconqueue(){
  ConQueue<PySample *> * obj = new ConQueue<PySample *>(10);
}
