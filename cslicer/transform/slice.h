#pragma once
#include <vector>
#include "graph/sample.h"
#include "graph/sliced_sample.h"

enum POLICY {
  PUSH,
  PULL,
  LOCAL
};

struct gpu{
   int cost[4];
};

class Slice{

  std::vector<int> workload_map;
  std::vector<int> storage_map[4];
  // Used for new node ordering
  int gpu_capacity[4];
  DuplicateRemover *dr;
  bool self_edge;
  int rounds;
public:

  Slice(std::vector<int> workload_map,
      std::vector<int> storage[4], bool self_edge, int rounds){
    this->workload_map = workload_map;
    int num_nodes = this->workload_map.size();

    for(int j = 0; j < num_nodes; j++){
      #pragma unroll
      for(int i=0;i<4;i++){
        this->storage_map[i].push_back(-1);
      }
    }
    for(int i=0;i<4;i++){
      int count = 0;
      for(auto j:storage[i]){
          this->storage_map[i][j] = count;
          count ++ ;
      }
      gpu_capacity[i] = count;
      std::cout << "gpu" << i <<"contains" << count <<"\n";
    }
    dr = new DuplicateRemover(this->workload_map.size());
    this->self_edge = self_edge;
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  void slice_layer(vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id,
            vector<POLICY> &policy);

  void measure_pull_benefits(Sample &s);

  void get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id, int num_layers);

  void  reorder(PartitionedLayer &l);

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
