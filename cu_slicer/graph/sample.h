#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <util/cuda_utils.h>
using namespace std;
// A simple sample structure
// The zeroth block is constructed from the set of target nodes as follows
// [null, null, batch_ids]
// Each block is constructed by sampling the neighbours of the previous blocks layer_nids.
struct Block{
  thrust::device_vector<long> offsets;
  thrust::device_vector<long> indices;
  thrust::device_vector<long> layer_nds;
  thrust::device_vector<long> in_degree;

public:
  void clear(){
    offsets.clear();
    indices.clear();
    layer_nds.clear();
    in_degree.clear();
  }

  void debug(){
    debugVector<long>(offsets,"offsets");
    debugVector<long>(indices,"indices");
    debugVector<long>(layer_nds,"layer_nds");
    debugVector<long>(in_degree,"in_degree");
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
