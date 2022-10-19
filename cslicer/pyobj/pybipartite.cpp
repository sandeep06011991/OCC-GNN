#include "pyobj/pybipartite.h"
#include <pybind11/stl.h>
namespace py = pybind11;
// Clean up this code completely
PyBipartite::PyBipartite(BiPartite *bp){
    // std::cout << bp->gpu_id <<"\n";
    // std::cout << bp->in_nodes.size() <<"\n";
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    this->gpu_id = bp->gpu_id;
    num_in_nodes = bp->num_in_nodes;
    num_out_nodes = bp->num_out_nodes;

    in_nodes = torch::from_blob(bp->in_nodes.data(), {bp->in_nodes.size()}, opts);
    indptr = torch::from_blob(bp->indptr.data(), {bp->indptr.size()}, opts);
    expand_indptr = torch::from_blob(bp->expand_indptr.data(), {bp->expand_indptr.size()}, opts);
    missing_node_ids = torch::from_blob(bp->missing_node_ids.data(), \
      {bp->missing_node_ids.size()}, opts);
    cached_node_ids  = torch::from_blob(bp->cached_node_ids.data(), \
      {bp->cached_node_ids.size()}, opts);

    out_nodes = torch::from_blob(bp->out_nodes.data(), {bp->out_nodes.size()}, opts);
    owned_out_nodes = torch::from_blob(bp->owned_out_nodes.data(), {bp->owned_out_nodes.size()}, opts);
    indices = torch::from_blob(bp->indices.data(), {bp->indices.size()}, opts);
    assert(expand_indptr_v.size() == indices_v.size());
    indegree = torch::from_blob(bp->in_degree.data(),{bp->in_degree.size()},opts);

    for(int i=0;i<4;i++){
      from_ids.push_back(torch::from_blob(bp->from_ids[i].data(), {bp->from_ids[i].size()}, opts));
      to_ids.push_back(torch::from_blob(bp->to_ids[i].data(), {bp->to_ids[i].size()}, opts));
    }

    self_ids_in = torch::from_blob(bp->self_ids_in.data(), {bp->self_ids_in.size()}, opts);
    self_ids_out = torch::from_blob(bp->self_ids_out.data(), {bp->self_ids_out.size()}, opts);


}

PyBipartite::~PyBipartite(){
  //std::cout << "destroy bipartite\n";
}

PySample::PySample(PartitionedSample &s){
    for(int i=0;i<3;i++){
        auto all_bipartites = new std::vector<PyBipartite *>();
        for(int j=0; j<4; j++){
          // std::cout <<"Working on" << j <<"\n";
          // std::cout << s->layers[i].bipartite[j] <<"\n";
          auto bipartite = new PyBipartite(s.layers[i].bipartite[j]);
          all_bipartites->push_back(bipartite);
          if(i==0){
            in_nodes = in_nodes + s.layers[i].bipartite[j]->in_nodes.size();
          }
          if(i==2){
            out_nodes = out_nodes + s.layers[i].bipartite[j]->out_nodes.size();
            missing_node_ids.push_back(bipartite->missing_node_ids);
            cached_node_ids.push_back(bipartite->cached_node_ids);
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
