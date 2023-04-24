#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>
#include "../util/device_vector.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"

using namespace std;
// A simple sample structure
// The zeroth block is constructed from the set of target nodes as follows
// [null, null, batch_ids]
// Each block is constructed by sampling the neighbours of the previous blocks layer_nids.

namespace cuslicer {

struct Block{
  cuslicer::device_vector<NDTYPE> offsets;
  cuslicer::device_vector<NDTYPE> indices;
  cuslicer::device_vector<NDTYPE> layer_nds;
  cuslicer::device_vector<NDTYPE> in_degree;

public:
  void clear(){
    offsets.clear();
    indices.clear();
    layer_nds.clear();
    in_degree.clear();
  }

  void debug(){
    offsets.debug("offsets");
    indices.debug("indices");
    layer_nds.debug("layer_nds");
    in_degree.debug("in_degree");
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

  void debug(){
    std::cout << "Sample:\n";
    for(int i=0;i<num_layers + 1;i++){
      std::cout << "Block:" << i <<"\n";
      block[i]->debug();
    }
  }
};

}
