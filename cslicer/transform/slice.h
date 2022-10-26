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
  bool self_edge;
public:

  Slice(std::vector<int> workload_map,
      std::vector<int> storage_map[4], bool self_edge){
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
    this->self_edge = self_edge;
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  void slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id);
};

void check_allocation_for_optimality(Sample &s, int *workload_map, long num_nodes);

struct redundant{
  int total_computation = 0;
  int redundant_computation = 0;
  int total_communication = 0;
  int redundant_communication = 0;
};


struct gpu_meta{
  int set_partitions;
  int partitions[4];
};


void populate_meta_dict();

redundant  print_statistics(Sample &s, std::vector<int> ** layer_color, long num_nodes, std::vector<int> & workload_map);
// redundant  print_statistics(Sample &s, vector<int>** layer_color, long num_nodes, vector<int>& workload_map){
