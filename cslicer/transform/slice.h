#pragma once
#include <vector>
#include "graph/sample.h"
#include "graph/sliced_sample.h"

class Slice{

  std::vector<int> workload_map;
  std::vector<int> storage_map[4];
  // Used for new node ordering
  int gpu_capacity[4];
  DuplicateRemover *dr;
public:

  Slice(std::vector<int> workload_map,
      std::vector<int> storage_map[4]){
    this->workload_map = workload_map;
    #pragma unroll
    for(int i=0;i<4;i++){
      this->storage_map[i] = storage_map[i];
    }
    for(int i=0;i<4;i++){
      int count = 0;
      for(auto j:this->storage_map[i]){
        if(j != -1){
          count ++ ;
        }
      }
      gpu_capacity[i] = count;
    }
    dr = new DuplicateRemover(this->workload_map.size());
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  void slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id);
};
