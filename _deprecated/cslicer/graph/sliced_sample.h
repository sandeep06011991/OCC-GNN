#pragma once
#include <vector>
#include <cassert>
#include "bipartite.h"

class PartitionedLayer{
  public:
    int num_partitions = 8;
    BiPartite* bipartite[8];
    PartitionedLayer(){
      // this->bipartite = (BiPartite **)malloc(sizeof(BiPartite *) * 4);
      this->num_partitions = num_partitions;
      for(int i=0;i<num_partitions;i++){
        this->bipartite[i] = new BiPartite(i);
      }
    }

    void clear(){
      for(int i=0;i<this->num_partitions;i++){
        this->bipartite[i]->refresh();
      }
    }

    void debug(){
        for(int i=0;i<this->num_partitions;i++){
          std::cout << "Bipartite graph" << i << "\n";
          bipartite[i]->debug();
        }
    }
    ~PartitionedLayer(){
       for(int i=0;i<this->num_partitions;i++){
         delete this->bipartite[i];
       }
    }
};

class PartitionedSample{
public:
  int num_layers = 5;
  int num_partitions = 4;
  // Fixed max layers == 5
  // Can be made a compile time constant. Do later.
  // Can allocate max possible layers or use compiler directives.
  PartitionedLayer layers[5];

  // From ids are storage order ids in the local cache or local feature
  // To ids are the position they are moved to in the input tensor
  std::vector<long> cache_hit_from[8];
  std::vector<long> cache_hit_to[8];
  std::vector<long> cache_miss_from[8];
  std::vector<long> cache_miss_to[8];
  // Nodes of the final raining values.
  std::vector<long> last_layer_nodes[8];

  PartitionedSample(int num_layers, int num_partitions){
    this->num_layers = num_layers;
    this->num_partitions = num_partitions;

  }

  void clear(){
    for(int i=0;i<num_layers;i++){
      layers[i].clear();
    }
    for(int i=0;i<this->num_partitions;i++){
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
