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
};

class PartitionedSample{
public:
  int num_layers = 3;
  // Fixed max layers == 4
  // Can be made a compile time constant. Do later.
  // Can allocate max possible layers or use compiler directives.
  PartitionedLayer layers[3];
  std::vector<long> refresh_map[4];
  std::vector<long> cache_hit[4];
  void clear(){
    for(int i=0;i<3;i++){
      layers[i].clear();
    }
    for(int i=0;i<4;i++){
      refresh_map[i].clear();
    }
  }

  void debug(){
    for(int i=0;i < 3; i++){
      layers[i].debug();
    }
  }
};
