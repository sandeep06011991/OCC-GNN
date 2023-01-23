#pragma once
#include "thrust/device_vector.h"
#include "graph/sample.h"
#include "graph/sliced_sample.h"


class Slice{

  thrust::device_vector<int> workload_map;
  thrust::device_vector<int> storage_map[8];
  thrust::device_vector<int> storage[8];
  // Used for new node ordering
  int gpu_capacity[8];
  int num_gpus = 4;
  DuplicateRemover *dr;
  // Use this for GAT
  bool self_edge = false;
  bool pull_optimization = false;
  bool use_map_for_duplicates = false;
  bool add_debug = false;

  int rounds;

public:
// Are all these options really needed.
  Slice(thrust::device_vector<int> workload_map,
      thrust::device_vector<int> storage[8], bool self_edge, int rounds,
        bool pull_optimization, int num_gpus){
    this->workload_map = workload_map;
    this->num_gpus= num_gpus;
    std::cout << "Setting number of gpus to" << num_gpus <<"\n";
    int num_nodes = this->workload_map.size();

    for(int j = 0; j < num_nodes; j++){
      #pragma unroll
      for(int i=0;i<this->num_gpus;i++){
        this->storage_map[i].push_back(-1);
      }

    }
    for(int i=0;i<num_gpus;i++){
      int count = 0;
      for(auto j:storage[i]){
          this->storage_map[i][j] = count;
          this->storage[i].push_back(j);
	         count ++ ;
      }
      gpu_capacity[i] = count;
    }
    this->rounds = rounds;
    dr = new ArrayMap(this->workload_map.size());
    this->self_edge = self_edge;
    std::cout << pull_optimization << "is in mode" <<"\n";
    this->pull_optimization = pull_optimization;
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  void slice_layer(thrust::device_vector<long>& in, Block &bl, PartitionedLayer& l, int layer_id);

  void measure_pull_benefits(Sample &s);

  // void get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id, int num_layers);

  void  reorder(PartitionedLayer &l);

  void debug_vector(string str, thrust::device_vector<long>& data, std::ostream& stream){
    stream << str <<":";
    int c = 0;
    for(long d: data){
      stream << d << " ";
      c ++ ;
      if (c % 20 == 0 ) stream <<  "\n";
    }
    stream <<"\n";
  }

};

void check_allocation_for_optimality(Sample &s, int *workload_map, long num_nodes);

struct redundant{
  int total_computation = 0;
  int redundant_computation = 0;
  int total_communication = 0;
  int redundant_communication = 0;

  void debug(){
    std::cout << "total_communication" << total_communication <<"\n";
    std::cout << "redundant_communication" << redundant_communication <<"\n";
    std::cout << "total_computation" << total_computation <<"\n";
    std::cout << "redundant_computation" << redundant_communication <<"\n";
  }
};


struct gpu_meta{
  int set_partitions;
  int partitions[8];
};


void populate_meta_dict();

redundant  print_statistics(Sample &s, thrust::device_vector<int> ** layer_color, long num_nodes,\
          thrust::device_vector<int> & workload_map,thrust::device_vector<int> storage_map[8]);
// redundant  print_statistics(Sample &s, vector<int>** layer_color, long num_nodes, vector<int>& workload_map){
