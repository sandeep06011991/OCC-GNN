#pragma once
#include "../util/device_vector.h"
#include "../graph/sample.h"
#include "../graph/sliced_sample.h"
#include "../util/types.h"
#include <curand.h>
#include <curand_kernel.h>

#include "balancer.cuh"
// Ogbn-products sample charectersitcs
// Nodes [1705194, 684535, 76689, 4096] Edges [12975036, 1490753, 80134]
// Reorder - map 
// Nodes [753565, 226407, 36803, 4096] Edges [1743108, 308245, 33447]

namespace cuslicer{

 __inline__ __device__
bool is_selected(long *id, size_t sz){
   if(sz == 0)return id[0] != 0;
   return id[sz] != id[sz-1];
}

  // This wierd data structure is needed as 
  // all the edges for the local bipartite graphs are constructed in one pass.
  // The samples are read in one kernel and the following kernel constructs
  // all the local bipartite graph. 
  // We do this by maintaining index for all gpu specfic data in one long 
  // array where each gpu data is stacked next to each other
  struct Vector_From_Index{
      long * data;
      long offset;

      __device__
      inline void  add_position_offset(long val, long pos){
          assert(pos - 1 -offset >= 0);
          // -1 because positions are caclulated from inclusie sum
          data[pos - 1 - offset ] = val;
      }
      // Used mostly by indptr. 
      
      __device__
      inline void  add_value_offset(long val, long pos){
        assert(pos >= 0);
        data[pos] = val - offset;
      }
  };

    struct LocalGraphInfo{
        Vector_From_Index in_nodes_local;
        Vector_From_Index in_nodes_pulled;
        Vector_From_Index out_nodes_local;
        Vector_From_Index out_nodes_remote;
        Vector_From_Index out_degree_local;
        Vector_From_Index indptr_L;
        Vector_From_Index indptr_R;
        long num_out_local;
        long num_in_local;
        long num_in_pulled;
        Vector_From_Index indices_L;
        Vector_From_Index indices_R;
        Vector_From_Index push_from_ids[MAX_GPUS];
        Vector_From_Index pull_to_ids[MAX_GPUS - 1];
    };


class Slice{
protected:
  device_vector<PARTITIONIDX> sample_workload_map;
  device_vector<PARTITIONIDX> workload_map;
  device_vector<NDTYPE> storage_map[8];
  device_vector<NDTYPE> storage[8];
  // device_vector<int> sample_partition;
  // Used for new node ordering
  void** storage_map_flattened;
  int gpu_capacity[8];
  int num_gpus = 4;
  DuplicateRemover *dr;
  // Use this for GAT
  bool pull_optimization = false;
  device_vector<NDTYPE> cache_hit_mask;
  device_vector<NDTYPE> cache_miss_mask;
  long num_nodes  = 0;
  cudaEvent_t event1;
  cudaEvent_t event2;
  cudaEvent_t event3;
  cudaEvent_t event4;
  cudaEvent_t event5;
  cudaEvent_t event6;
  cudaEvent_t event7;

  curandState * rand_states;
  const int TOTAL_RAND_STATES = MAX_BLOCKS * BLOCK_SIZE;

  LocalGraphInfo host_graph_info[MAX_GPUS];
  LocalGraphInfo * device_graph_info;

  LoadBalancer *loadbalancer;
public:
// Are all these options really needed.
  Slice(device_vector<PARTITIONIDX> workload_map,
      std::vector<NDTYPE> storage[8],
        bool pull_optimization, int num_gpus, curandState *s){
    gpuErrchk(cudaEventCreate(&event1));
    gpuErrchk(cudaEventCreate(&event2));
    gpuErrchk(cudaEventCreate(&event3));
    gpuErrchk(cudaEventCreate(&event4));
    gpuErrchk(cudaEventCreate(&event5));
    gpuErrchk(cudaEventCreate(&event6));
    gpuErrchk(cudaEventCreate(&event7));
    this->loadbalancer = new LoadBalancer(num_gpus, s, TOTAL_RAND_STATES);
    this->rand_states = s;
    this->workload_map = workload_map;
    this->num_gpus= num_gpus;
    long num_nodes = this->workload_map.size();
    this->num_nodes = num_nodes;
    assert(num_gpus <= 8);
    std::vector<NDTYPE> _t1;
    std::vector<NDTYPE> _t2;
    for(int i=0;i<num_gpus;i++){
      _t1.clear();
      _t2.clear();
      for(int j = 0; j < num_nodes; j++){
          _t1.push_back(-1);
      }
      _t2 = storage[i];
      int count  = 0;
      for(auto j:_t2){
          _t1[j] = count;
           count ++ ;
      }
      // Must be an lvalue
      auto _s1 = device_vector<NDTYPE>(_t1);
      this->storage_map[i] = _s1;
      auto s2 = device_vector<NDTYPE>(_t2);
      this->storage[i] = s2;
      gpu_capacity[i] = count;
    }
    void *t[8];
    gpuErrchk(cudaMalloc(&storage_map_flattened, num_gpus * sizeof(int *)));
    for(int i= 0; i < num_gpus; i++){
    	t[i] = this->storage_map[i].ptr();
      // std::cout << "Storage map size" << this->storage_map[i].size() <<"\n";
    }
    std::cout << "All storage maps moved\n";

    gpuErrchk(cudaMemcpy(storage_map_flattened, t, sizeof(int *) * num_gpus,\
      cudaMemcpyHostToDevice));

    dr = new ArrayMap(this->workload_map.size());
    this->pull_optimization = pull_optimization;
  }

  void slice_sample(Sample &s, PartitionedSample &ps);

  virtual void slice_layer(device_vector<long>& in, Block &bl, \
      PartitionedLayer& l, bool last_layer) = 0;


  // void get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id, int num_layers);

  void  reorder(PartitionedLayer &l) ;

  void fill_cache_hits_and_misses(PartitionedSample &ps, int gpu, device_vector<long> &in_nodes);

  virtual void resize_bipartite_graphs(PartitionedLayer &ps,int num_in_nodes, int num_out_nodes,\
    int num_edges) = 0;

};

class PushSlicer: public Slice{


    // Contains information from the offsets array and exclusive sum.
    // Use to construct graphs from partitioned edges.
    // Must have one to one mapping from every object in bipartite graph
    
public:
    PushSlicer(device_vector<PARTITIONIDX> workload_map,
        std::vector<NDTYPE> storage[8],
          bool pull_optimization, int num_gpus, curandState *state):Slice(workload_map,
            storage, pull_optimization, num_gpus, state){
              gpuErrchk(cudaMalloc(&device_graph_info, sizeof(LocalGraphInfo) * this->num_gpus ));
    }

    void copy_graph_info(){
      gpuErrchk(cudaMemcpy(device_graph_info, host_graph_info,  sizeof(LocalGraphInfo) * this->num_gpus, cudaMemcpyHostToDevice));
    }

    void slice_layer(device_vector<NDTYPE>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer) ;

    void slice_layer_per_gpu(device_vector<NDTYPE>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer, int gpu);

    void resize_bipartite_graphs(PartitionedLayer &ps,int num_in_nodes, int num_out_nodes,
        int num_edges);
};

class PullSlicer: public Slice{

    
public:
    PullSlicer(device_vector<PARTITIONIDX> workload_map,
        std::vector<NDTYPE>  storage[8],
          bool pull_optimization, int num_gpus, curandState *state):Slice(workload_map,
            storage, pull_optimization, num_gpus, state){
              gpuErrchk(cudaMalloc(&device_graph_info, sizeof(LocalGraphInfo) * this->num_gpus ));
            }

    void slice_layer(device_vector<NDTYPE>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer);

    void resize_bipartite_graphs(PartitionedLayer &ps,
      int num_in_nodes, int num_out_nodes,
          int num_edges);    

    void copy_graph_info(){
      gpuErrchk(cudaMemcpy(device_graph_info, host_graph_info,  sizeof(LocalGraphInfo) * this->num_gpus, cudaMemcpyHostToDevice));
    }

};


}
