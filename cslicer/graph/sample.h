#pragma once
#include <cstdlib>
#include "spdlog/spdlog.h"

// A simple sample structure
// The zeroth block is constructed from the set of target nodes as follows
// [null, null, batch_ids]
// Each block is constructed by sampling the neighbours of the previous blocks layer_nids.
struct Block{
  std::vector<long> offsets;
  std::vector<long> indices;
  std::vector<long> layer_nds;
  std::vector<long> in_degree;

public:
  void clear(){
    offsets.clear();
    indices.clear();
    layer_nds.clear();
    in_degree.clear();
  }
};

class Sample{
public:
  Block ** block;
  // num layers is +1 as the zero the layer is the training nodes.
  // Iterate over this from [1, layer + 1)
  int num_layers;
  Sample(int num_layers){
    block = (Block **) malloc(sizeof(Block) * num_layers + 1);
    for(int i=0;i<num_layers+1;i++){
      block[i] = new Block();
    }
    this->num_layers = num_layers;
  }
  void clear(){
    for(int i=0;i<num_layers;i++){
      block[i]->clear();
    }
  }
};
