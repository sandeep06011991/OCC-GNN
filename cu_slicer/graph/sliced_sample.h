#pragma once
#include <vector>
#include <cassert>
#include "bipartite.h"

class PartitionedLayer{
  public:
    BiPartite* bipartite[4];
    PartitionedLayer(){
      // this->bipartite = (BiPartite **)malloc(sizeof(BiPartite *) * 4);
      for(int i=0;i<4;i++){
        this->bipartite[i] = new BiPartite(i);
      }
    }

    void clear(){
      for(int i=0;i<4;i++){
        this->bipartite[i]->refresh();
      }
    }

    void debug(){
        for(int i=0;i<4;i++){
          std::cout << "Bipartite graph" << i << "\n";
          bipartite[i]->debug();
        }
    }
    ~PartitionedLayer(){
       for(int i=0;i<4;i++){
         delete this->bipartite[i];
       }
    }
};

class PartitionedSample{
public:
  int num_layers = 3;
  // Fixed max layers == 4
  // Can be made a compile time constant. Do later.
  // Can allocate max possible layers or use compiler directives.
  PartitionedLayer layers[4];

  // From ids are storage order ids in the local cache or local feature
  // To ids are the position they are moved to in the input tensor
  std::vector<long> cache_hit_from[4];
  std::vector<long> cache_hit_to[4];
  std::vector<long> cache_miss_from[4];
  std::vector<long> cache_miss_to[4];
  // Nodes of the final raining values.
  std::vector<long> last_layer_nodes[4];

  PartitionedSample(int num_layers){
    this->num_layers = num_layers;
  }

  void clear(){
    for(int i=0;i<num_layers;i++){
      layers[i].clear();
    }
    for(int i=0;i<4;i++){
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
