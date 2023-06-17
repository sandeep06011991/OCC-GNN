#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;
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

  void debug(){
    std::cout <<"offsets:";
    for(auto nd: offsets){
      std::cout << nd <<  " ";
    }

      std::cout <<"\nindices:";
    for(auto nd: indices){
      std::cout << nd <<  " ";
    }
      std::cout <<"\nlayer nds:";
    for(auto nd: layer_nds){
      std::cout << nd <<  " ";
    }
    std::cout <<"\nin degree:";
    for(auto nd: in_degree){
      std::cout << nd <<  " ";
    }
    std::cout << "\n";

  }

  void check_remote(vector<int> &map, vector<long>& dest){
  	assert(false);
	//Not migrated.
	  int local_edge[4];
	int remote_edge[4];
	for(int i=0;i<4;i++){
		local_edge[i] = 0;
		remote_edge[i] = 0;

	}

	for(int i=0;i<offsets.size()-1;i++){
		long dest_nd = dest[i];
		long dst_p = map[dest_nd];
		for(int j=offsets[i] ; j <offsets[i+1];j++){
			long src_nd = indices[j];
			long src_p = map[src_nd];
			if(src_p == dst_p){
				local_edge[src_p] ++;
			}else{
				remote_edge[src_p] ++;
			}
		}
	}
	for(int i = 0;i<4;i++){
	
		std::cout << "patition" << i <<":" << local_edge[i] <<":"<<remote_edge[i] <<"\n";
	}
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

  void check_remote(vector<int> &p_map){
  	for(int i=1;i<num_layers+1;i++){
		block[i]->check_remote(p_map,block[i-1]->layer_nds); 
	}
  }
  void debug(){
    for(int i=0;i<num_layers + 1;i++){
      block[i]->debug();
    }
  }
};
