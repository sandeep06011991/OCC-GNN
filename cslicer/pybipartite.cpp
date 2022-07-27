#include "pybipartite.h"
// #include "sample.h"
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <iostream>
// #include "util/conqueue.h"
#include <pybind11/stl.h>
namespace py = pybind11;

PyBipartite::PyBipartite(BiPartite *bp){
    // std::cout << bp->gpu_id <<"\n";
    // std::cout << bp->in_nodes.size() <<"\n";
    missing_node_ids = bp->missing_node_ids;
    this->gpu_id = bp->gpu_id;
    in_nodes_v = bp->in_nodes;
    data.clear();
    in_nodes_start = data.size();
    in_nodes_end = data.size() + in_nodes_v.size();
    data.insert(data.end(),in_nodes_v.begin(), in_nodes_v.end());
    auto opts = torch::TensorOptions().dtype(torch::kLong);
    in_nodes = torch::from_blob(in_nodes_v.data(), {in_nodes_v.size()}, opts);

    indptr_v = bp->indptr;
    indptr = torch::from_blob(indptr_v.data(), {indptr_v.size()}, opts);
    indptr_start = data.size();
    indptr_end = data.size() + indptr_v.size();
    data.insert(data.end(), indptr_v.begin(), indptr_v.end());

    expand_indptr_v = bp->expand_indptr;
    expand_indptr_start = data.size();
    expand_indptr_end = data.size() + bp->expand_indptr.size();
    data.insert(data.end(), expand_indptr_v.begin(), expand_indptr_v.end());
    expand_indptr = torch::from_blob(expand_indptr_v.data(), {expand_indptr_v.size()}, opts);

    num_in_nodes = bp->num_in_nodes;
    num_out_nodes = bp->num_out_nodes;
    out_nodes_v = bp->out_nodes;
    out_nodes_start = data.size();
    out_nodes_end = out_nodes_v.size() + data.size();
    data.insert(data.end(), out_nodes_v.begin(), out_nodes_v.end());
    out_nodes = torch::from_blob(out_nodes_v.data(), {out_nodes_v.size()}, opts);

    owned_out_nodes_v = bp->owned_out_nodes;
    owned_out_nodes = torch::from_blob(owned_out_nodes_v.data(), {owned_out_nodes_v.size()}, opts);
    owned_out_nodes_start = data.size();
    owned_out_nodes_end = owned_out_nodes_v.size() + data.size();
    data.insert(data.end(), owned_out_nodes_v.begin(), owned_out_nodes_v.end());

    indices_v = bp->indices;
    indices_start = data.size();
    indices_end = data.size() +  indices_v.size();
    data.insert(data.end(), indices_v.begin(), indices_v.end());
    indices = torch::from_blob(indices_v.data(), {indices_v.size()}, opts);
    assert(expand_indptr_v.size() == indices_v.size());

    indegree_v = bp->in_degree;
    indegree_start = data.size();
    indegree_end = data.size() + indegree_v.size();
    data.insert(data.end(), indegree_v.begin(),indegree_v.end());
    indegree = torch::from_blob(indegree_v.data(),{indegree_v.size()},opts);

    for(int i=0;i<4;i++){
      from_ids_start.push_back(data.size());
      from_ids_end.push_back(data.size() + bp->from_ids[i].size());
      from_ids_v.push_back(bp->from_ids[i]);
      data.insert(data.end(), bp->from_ids[i].begin(), bp->from_ids[i].end());
      from_ids.push_back(torch::from_blob(from_ids_v[i].data(), {from_ids_v[i].size()}, opts));

      to_ids_start.push_back(data.size());
      to_ids_end.push_back(data.size() + bp->to_ids[i].size());
      to_ids_v.push_back(bp->to_ids[i]);
      data.insert(data.end(), bp->to_ids[i].begin(), bp->to_ids[i].end());
      to_ids.push_back(torch::from_blob(to_ids_v[i].data(), {to_ids_v[i].size()}, opts));
    }
    self_ids_in_v = bp->self_ids_in;
    self_ids_in_start = data.size();
    self_ids_in_end = data.size() + bp->self_ids_in.size();
    data.insert(data.end(), self_ids_in_v.begin(), self_ids_in_v.end());

    self_ids_in = torch::from_blob(self_ids_in_v.data(), {self_ids_in_v.size()}, opts);
    self_ids_out_v = bp->self_ids_out;
    self_ids_out = torch::from_blob(self_ids_out_v.data(), {self_ids_out_v.size()}, opts);
    self_ids_out_start = data.size();
    self_ids_out_end = data.size() + bp->self_ids_out.size();
    data.insert(data.end(), self_ids_out_v.begin(), self_ids_out_v.end());

    data_tensor = torch::from_blob(data.data(),{data.size()}, opts);
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
    // data_tensor.share_memory_();
}

PyBipartite::~PyBipartite(){
  //std::cout << "destroy bipartite\n";
}

PySample::PySample(Sample *s){
    for(int i=0;i<3;i++){
        auto all_bipartites = new std::vector<PyBipartite *>();
        for(int j=0; j<4; j++){
          // std::cout <<"Working on" << j <<"\n";
          // std::cout << s->layers[i].bipartite[j] <<"\n";
          auto bipartite = new PyBipartite(s->layers[i].bipartite[j]);
          all_bipartites->push_back(bipartite);
          missing_node_ids.push_back(bipartite->missing_node_ids);
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
    //std::cout << "Code to check that destructor is being called\n";
}

void testconqueue(){
  ConQueue<PySample *> * obj = new ConQueue<PySample *>(10);
}
