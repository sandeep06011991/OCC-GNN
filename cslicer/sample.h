#pragma once
#include <vector>
#include <cassert>
#include "bipartite.h"

class Layer{
  public:
    BiPartite** bipartite;
    Layer(){
      this->bipartite = (BiPartite **)malloc(sizeof(BiPartite) * 4);
      for(int i=0;i<4;i++){
        this->bipartite[i] = new BiPartite(i);
      }
    }

    void clear(){
      for(int i=0;i<4;i++){
        this->bipartite[i]->refresh();
      }
    }
};

class Sample{
public:
  int num_layers;
  // Fixed max layers == 4
  // Can be made a compile time constant. Do later.
  Layer layers[3];
  std::vector<long> refresh_map[4];
  static Sample * get_dummy_sample(){
    return new Sample();
  }
};
