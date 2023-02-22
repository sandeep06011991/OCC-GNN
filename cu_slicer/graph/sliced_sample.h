#pragma once
#include <vector>
#include <cassert>
#include "bipartite.h"
#include "../util/device_vector.h"
#include "../util/cuda_utils.h"

namespace cuslicer{

class PartitionedLayer{
  public:
    BiPartite* bipartite[8];

    // From each layers graph we select edges per each gpu.
    // The index_* i th element is set to one to indicate that it is selected
    // We need one index_* per element to be partitioned per gpu
    // To minimize overhead we squash all index_* per gpu into the same vector for all gpus.
    device_vector<long> index_in_nodes;
    device_vector<long> index_out_nodes_local;
    device_vector<long> index_out_nodes_remote;
    device_vector<long> index_indptr_local;
    device_vector<long> index_indptr_remote;
    device_vector<long> index_edge_local;
    device_vector<long> index_edge_remote;

    int num_gpus = -1;
    PartitionedLayer(){}

    void set_number_of_gpus(int num_gpus){
      this->num_gpus = num_gpus;
      // this->bipartite = (BiPartite **)malloc(sizeof(BiPartite *) * 4);
      for(int i=0; i<num_gpus; i++){
        this->bipartite[i] = new BiPartite(i, num_gpus);
      }
    }

    // Correct sizes are calculated based on partitioning strategy
    // Push requires num_out_nodes = sample_out_nodes * num_gpus * num_gpus
    // Pull resuires num_in_nodes = sample_in_nodes * num_gpus * num_gpus
    void resize_selected_push(int num_out_nodes_local, int num_out_nodes_remote,\
             int num_edge_local, int num_edge_remote,  int num_in_nodes){
       // n refers to number of graph partitions
       // Sizes per partition are given from slicer.
       // eg. num_out_nodes_local = nodes in sample out * N_GPUS
       index_in_nodes.resize_and_zero(num_in_nodes);
       index_out_nodes_local.resize_and_zero(num_out_nodes_local);
       index_out_nodes_remote.resize_and_zero(num_out_nodes_remote);
       index_indptr_local.resize_and_zero(num_out_nodes_local);
       index_indptr_remote.resize_and_zero(num_out_nodes_remote);
       index_edge_local.resize_and_zero(num_edge_local);
       index_edge_remote.resize_and_zero(num_edge_remote);
   }

  void debug_index(){
    std::cout << "Partitioned Sample:\n";
    index_in_nodes.debug("index_in_nodes");
    index_out_nodes_local.debug("index out nodes local");
    index_out_nodes_remote.debug("index_out_nodes remote");
    index_indptr_local.debug("index_indptr_local");
    index_indptr_remote.debug("index indptr remote");
    index_edge_local.debug("index edge local");
    index_edge_remote.debug("index edge remote");
  }

  //
  //   void inclusive_scan_indptr(long * local_nodes){
  //     for(int dest=0;dest<this->num_gpus;dest++){
  //       for(int src=0;src<this->num_gpus;src++){
  //         long N = local_nodes[this->num_gpus * dest + src];
  //         if(N != 0){
  //           thrust::device_vector<long> & indptr = bipartite[dest]->indptr_[src];
  //           thrust::inclusive_scan(indptr.begin(), indptr.end(), indptr.begin());
  //         }
  //       }
  //     }
  //   }

    void clear(){
      for(int i=0;i<this->num_gpus;i++){
        this->bipartite[i]->refresh();
      }
    }

    void debug(){
        for(int i=0;i<this->num_gpus;i++){
          std::cout << "Bipartite graph" << i << "\n";
          bipartite[i]->debug();
        }
    }
    ~PartitionedLayer(){
       for(int i=0;i<this->num_gpus;i++){
         delete this->bipartite[i];
       }
    }
};

class PartitionedSample{
public:
  int num_layers = MAX_LAYERS;
  // Fixed max layers == 4
  // Can be made a compile time constant. Do later.
  // Can allocate max possible layers or use compiler directives.
  PartitionedLayer layers[MAX_DEVICES];

  // From ids are storage order ids in the local cache or local feature
  // To ids are the position they are moved to in the input tensor
  device_vector<long> cache_hit_from[MAX_DEVICES];
  device_vector<long> cache_hit_to[MAX_DEVICES];
  device_vector<long> cache_miss_from[MAX_DEVICES];
  device_vector<long> cache_miss_to[MAX_DEVICES];
  // Nodes of the final raining values.
  device_vector<long> last_layer_nodes[MAX_DEVICES];
  int num_gpus = -1;

  PartitionedSample(int num_layers, int num_gpus){
    this->num_layers = num_layers;
    this->num_gpus = num_gpus;
    for(int i=0;i < this->num_layers; i ++){
      layers[i].set_number_of_gpus(this->num_gpus);
    }

  }

  void clear(){
    for(int i=0;i<num_layers;i++){
      layers[i].clear();
    }
    for(int i=0;i<num_gpus;i++){
      cache_hit_from[i].clear();
      cache_hit_to[i].clear();
      cache_miss_from[i].clear();
      cache_miss_to[i].clear();
    }
  }

  void debug(){
    for(int i=0;i < num_layers; i++){
      std::cout << "Layer" << i <<"\n";
      layers[i].debug();

    }
  }
};

}
