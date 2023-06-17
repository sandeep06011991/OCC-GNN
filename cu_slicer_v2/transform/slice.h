#pragma once
#include "../util/device_vector.h"
#include "../graph/sample.h"
#include "../graph/sliced_sample.h"
#include "../util/types.h"
#include "../graph/order_book.h"
#include <curand.h>
#include <curand_kernel.h>
#include <memory>
#include "balancer.cuh"
// Ogbn-products sample charectersitcs
// Nodes [1705194, 684535, 76689, 4096] Edges [12975036, 1490753, 80134]
// Reorder - map 
// Nodes [753565, 226407, 36803, 4096] Edges [1743108, 308245, 33447]

namespace cuslicer{

 __inline__ __device__
bool is_selected(NDTYPE  *id, size_t sz){
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
      NDTYPE * data;
      long offset;

      __device__
      inline void  add_position_offset(NDTYPE val, long pos){
          if((pos - 1 -offset < 0)){
            printf("assert fail pos offset %ld %ld %ld\n", pos, offset,val);
          }
          if (val < 0){
            printf("adding 0 at position  %ld offset %ld val %ld\n", pos, offset, val);
          }
          assert(pos - 1 -offset >= 0);
          assert(val != -1);
          
          // -1 because positions are caclulated from inclusie sum
          data[pos - 1 - offset ] = val;
      }
      // Used mostly by indptr. 
      
      __device__
      inline void  add_value_offset(NDTYPE val, long pos){
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
  int num_gpus = -1;
  // Contains all information regarding partitioning and ordering
  std::shared_ptr<OrderBook> orderbook;
  device_vector<PARTITIONIDX> sample_workload_map;
  DuplicateRemover *dr;
  // Use this for GAT
  
  device_vector<NDTYPE> cache_hit_mask;
  device_vector<NDTYPE> cache_miss_mask;
  NDTYPE num_nodes  = 0;
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
  Slice(std::shared_ptr<OrderBook> orderbook, \
        int num_gpus, curandState *s, NDTYPE num_nodes){
    std::cout << "Do I need all these events \n";
    gpuErrchk(cudaEventCreate(&event1));
    gpuErrchk(cudaEventCreate(&event2));
    gpuErrchk(cudaEventCreate(&event3));
    gpuErrchk(cudaEventCreate(&event4));
    gpuErrchk(cudaEventCreate(&event5));
    gpuErrchk(cudaEventCreate(&event6));
    gpuErrchk(cudaEventCreate(&event7));
    this->loadbalancer = new LoadBalancer(num_gpus, s, TOTAL_RAND_STATES);
    this->rand_states = s;
    this->orderbook = orderbook;
    this->num_gpus= num_gpus;
    this->num_nodes = num_nodes;
    assert(num_gpus <= 8);

    dr = new ArrayMap(num_nodes);
    
  }

  void slice_sample(Sample &s, PartitionedSample &ps, bool loadbalancing);

  virtual void slice_layer(device_vector<NDTYPE>& in, Block &bl, \
      PartitionedLayer& l, bool last_layer) = 0;


  // void get_edge_policy(vector<long> &in, Block &bl, vector<POLICY> &policy, int layer_id, int num_layers);

  void  reorder(PartitionedLayer &l) ;

  void fill_cache_hits_and_misses(PartitionedSample &ps, \
        int gpu, device_vector<NDTYPE> &in_nodes);
  virtual void resize_bipartite_graphs(PartitionedLayer &ps,int num_in_nodes, int num_out_nodes,\
    int num_edges) = 0;

};

class PushSlicer: public Slice{


    // Contains information from the offsets array and exclusive sum.
    // Use to construct graphs from partitioned edges.
    // Must have one to one mapping from every object in bipartite graph
    
public:
    PushSlicer(std::shared_ptr<OrderBook> orderbook, \
        int num_gpus,curandState *state, int num_nodes):Slice(orderbook, num_gpus, state, num_nodes){
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
    PullSlicer(std::shared_ptr<OrderBook> orderbook, \
        int num_gpus,curandState *state, NDTYPE num_nodes):Slice(orderbook, num_gpus, state, num_nodes){
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

class FusedSlicer: public PullSlicer{
  public: 
    FusedSlicer(std::shared_ptr<OrderBook> orderbook, \
        int num_gpus,curandState *state, NDTYPE num_nodes):PullSlicer(orderbook, num_gpus, state, num_nodes){
            }
    void slice_layer(device_vector<NDTYPE>& in, Block &bl, \
        PartitionedLayer& l, bool last_layer);

};


}
