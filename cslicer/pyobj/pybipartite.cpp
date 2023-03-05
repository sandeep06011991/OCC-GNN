#include "pyobj/pybipartite.h"
#include <pybind11/stl.h>
namespace py = pybind11;


PyBipartite::PyBipartite(BiPartite *bp){
    // std::cout << bp->gpu_id <<"\n";
    // std::cout << bp->in_nodes.size() <<"\n";
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    this->gpu_id = bp->gpu_id;
    num_in_nodes_local = bp->num_in_nodes_local;
    num_in_nodes_pulled = bp->num_in_nodes_pulled;
    num_out_local = bp->num_out_local;
    num_out_remote = bp->num_out_remote;

    indptr_L = torch::from_blob(bp->indptr_L.data(), {bp->indptr_L.size()}, opts);
    indices_L = torch::from_blob(bp->indices_L.data(), {bp->indices_L.size()}, opts);

    indptr_R = torch::from_blob(bp->indptr_R.data(), {bp->indptr_R.size()}, opts);
    indices_R = torch::from_blob(bp->indices_R.data(), {bp->indices_R.size()}, opts);
    out_degree_local = torch::from_blob(bp->out_degree_local.data(), {bp->out_degree_local.size()}, opts);

    to_offsets.push_back(bp->to_offsets[0]);
    pull_from_offsets.push_back(bp->pull_from_offsets[0]);
    for(int i=0;i<4;i++){
      to_offsets.push_back(bp->to_offsets[i+1]);
      pull_from_offsets.push_back(bp->pull_from_offsets[i+1]);
      from_ids.push_back(torch::from_blob(bp->from_ids[i].data(), {bp->from_ids[i].size()}, opts));
      push_to_ids.push_back(torch::from_blob(bp->push_to_ids[i].data(), {bp->push_to_ids[i].size()}, opts));
    }

    self_ids_offset = bp->self_ids_offset;

}

PyBipartite::~PyBipartite(){
  //std::cout << "destroy bipartite\n";
}

PySample::PySample(PartitionedSample &s){
    for(int i=0;i<s.num_layers;i++){
        auto all_bipartites = new std::vector<PyBipartite *>();
        for(int j=0; j<4; j++){
          auto bipartite = new PyBipartite(s.layers[i].bipartite[j]);
          all_bipartites->push_back(bipartite);
          }
        layers.push_back(all_bipartites);
    }
      auto opts = torch::TensorOptions().dtype(torch::kInt64);
    for(int i=0;i<4;i++){
      cache_hit_from.push_back(torch::from_blob(s.cache_hit_from[i].data(), {s.cache_hit_from[i].size()}, opts));
      cache_hit_to.push_back(torch::from_blob(s.cache_hit_to[i].data(), {s.cache_hit_to[i].size()}, opts));
      cache_miss_from.push_back(torch::from_blob(s.cache_miss_from[i].data(), {s.cache_miss_from[i].size()}, opts));
      cache_miss_to.push_back(torch::from_blob(s.cache_miss_to[i].data(), {s.cache_miss_to[i].size()}, opts));
      out_nodes.push_back(torch::from_blob(s.last_layer_nodes[i].data(), {s.last_layer_nodes[i].size()}, opts));
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
