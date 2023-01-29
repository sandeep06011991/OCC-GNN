#pragma once
#include "thrust/device_vector.h"
#include "graph/sample.h"
#include "graph/sliced_sample.h"


class Slice{

  thrust::device_vector<int> workload_map;
  thrust::device_vector<int> storage_map[8];
  thrust::device_vector<int> storage[8];
  // Used for new node ordering
  void** storage_map_flattened;
  int gpu_capacity[8];
  int num_gpus = 4;
  DuplicateRemover *dr;
  // Use this for GAT
  bool pull_optimization = false;
  thrust::device_vector<int> cache_hit_mask;
  thrust::device_vector<int> cache_miss_mask;
  long num_nodes  = 0;
public:
// Are all these options really needed.
  Slice(thrust::device_vector<int> workload_map,
      thrust::device_vector<int> storage[8],
        bool pull_optimization, int num_gpus){
          std::cout<<"CREATING SLICER XXXXXXX\n";
    this->workload_map = workload_map;
    this->num_gpus= num_gpus;
    long num_nodes = this->workload_map.size();
    this->num_nodes = num_nodes;
    assert(num_gpus <= 8);
    for(int j = 0; j < num_nodes; j++){
      #pragma unroll
      for(int i=0;i<this->num_gpus;i++){
        this->storage_map[i].push_back(-1);
      }
    }
    std::cout << "NUM GPUS" << this->num_gpus << "\n";
    for(int i=0;i<num_gpus;i++){
      int count  = 0;
      for(auto j:storage[i]){
          this->storage_map[i][j] = count;
          this->storage[i].push_back(j);
	         count ++ ;
      }
      gpu_capacity[i] = count;
    }
    void *t[8];
    gpuErrchk(cudaMalloc(&storage_map_flattened, num_gpus * sizeof(int *)));
    for(int i= 0; i < num_gpus; i++){
    	t[i] = thrust::raw_pointer_cast(this->storage_map[i].data());
      std::cout << "Storage map size" << this->storage_map[i].size() <<"\n";
    }
    gpuErrchk(cudaMemcpy(storage_map_flattened, t, sizeof(int *) * num_gpus,\
      cudaMemcpyHostToDevice));

    dr = new ArrayMap(this->workload_map.size());
    this->pull_optimization = pull_optimization;
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  void slice_layer(thrust::device_vector<long>& in, Block &bl, \
      PartitionedLayer& l, bool last_layer);



  // void get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id, int num_layers);

  void  reorder(PartitionedLayer &l);

  void fill_cache_hits_and_misses(PartitionedSample &ps, int gpu, thrust::device_vector<long> &in_nodes);


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
