#include "pyobj/pybipartite.h"
#include <pybind11/stl.h>
#include "../graph/bipartite.h"
#include <thrust/device_vector.h>
namespace py = pybind11;

using namespace cuslicer;

PyBipartite::PyBipartite(BiPartite *bp, int local_gpu_id, int num_gpus){
    // std::cout << bp->gpu_id <<"\n";
    // std::cout << bp->in_nodes.size() <<"\n";
    auto opts = torch::TensorOptions().dtype(torch::kInt64)\
    .device(torch::kCUDA, local_gpu_id);

    this->gpu_id = bp->gpu_id;
    num_in_nodes_local = bp->num_in_nodes_local;
    num_in_nodes_pulled = bp->num_in_nodes_pulled;
    num_out_local = bp->num_out_local;
    num_out_remote = bp->num_out_remote;

    indptr_L = getTensor(bp->indptr_L, opts);
    indices_L = getTensor(bp->indices_L, opts);
    indptr_R = getTensor(bp->indptr_R, opts);
    indices_R = getTensor(bp->indices_R, opts);
    out_degree_local = getTensor(bp->out_degree_local, opts);

    to_offsets.push_back(bp->to_offsets[0]);
    pull_from_offsets.push_back(bp->pull_from_offsets[0]);
    for(int i=0;i<num_gpus; i++){
      to_offsets.push_back(bp->to_offsets[i+1]);
      pull_from_offsets.push_back(bp->pull_from_offsets[i+1]);
      from_ids.push_back(getTensor(bp->push_from_ids[i], opts));
      push_to_ids.push_back(getTensor(bp->pull_to_ids[i], opts));
    }

    self_ids_offset = bp->self_ids_offset;

}

PyBipartite::~PyBipartite(){
  //std::cout << "destroy bipartite\n";
}

PySample::PySample(PartitionedSample &s, int current_gpu, int num_gpus){
    for(int i=0;i<s.num_layers;i++){
        auto all_bipartites = new std::vector<PyBipartite *>();
        for(int j=0; j<num_gpus; j++){
          auto bipartite = new PyBipartite(s.layers[i].bipartite[j], current_gpu, num_gpus);
          all_bipartites->push_back(bipartite);
          }
        layers.push_back(all_bipartites);
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, current_gpu);
    for(int i=0;i<num_gpus;i++){
      cache_hit_from.push_back(getTensor(s.cache_hit_from[i], opts));
      cache_hit_to.push_back(getTensor(s.cache_hit_to[i], opts));
      cache_miss_from.push_back(getTensor(s.cache_miss_from[i], opts));
      cache_miss_to.push_back(getTensor(s.cache_miss_to[i], opts));
      out_nodes.push_back(getTensor(s.last_layer_nodes[i], opts));
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
