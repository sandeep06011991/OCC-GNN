#include "duplicate.h"
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "cuda_utils.h"
#include "cub.h"
// using namespace std;
using namespace cuslicer;


template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void set_nodes_not_present(NDTYPE * nodes, size_t nodes_size,
        NDTYPE * mask, size_t mask_size, NDTYPE *_tv){
    int tileId = blockIdx.x;
    int last_tile = ((nodes_size - 1) / TILE_SIZE + 1);
    while(tileId < last_tile){
    int start = threadIdx.x + (tileId * TILE_SIZE);
    int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),nodes_size);
    while(start < end){
      int id = start;
      #ifdef DEBUG
        if(nodes[id] >= mask_size){
          printf("node id %ld greater than mask %ld %ld \n", nodes[id], mask_size, nodes_size);
        }
          assert(nodes[id] < mask_size);
      #endif
      if(mask[nodes[id]] == 0){
            _tv[id] = 1;
           }else{
           _tv[id] = 0;
       }
       start += BLOCK_SIZE;
    }
    tileId += gridDim.x;
  }
}


template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void get_unique_nodes(NDTYPE*nodes, size_t nodes_size,
      NDTYPE *mask, size_t mask_size,
        NDTYPE*_tv, NDTYPE*_tv1, size_t tv1_size){
          int tileId = blockIdx.x;
          int last_tile = ((nodes_size - 1) / TILE_SIZE + 1);
          while(tileId < last_tile){
          int start = threadIdx.x + (tileId * TILE_SIZE);
          int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),nodes_size);

    while(start < end){
      int id = start;
    #ifdef DEBUG
        assert(nodes[id] < mask_size);
    #endif
    if(mask[nodes[id]] == 0){
      #ifdef DEBUG
        assert(_tv[id] < tv1_size);
      #endif
      _tv1[_tv[id]] = nodes[id];
      }
       start += BLOCK_SIZE;
   }
      tileId += gridDim.x;
  }
}



template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void update_mask_with_unique(NDTYPE *mask, size_t mask_size,
      int current_unique_nodes, NDTYPE* _tv1, size_t sz){
    int tileId = blockIdx.x;
    int last_tile = ((sz - 1) / TILE_SIZE + 1);
    while(tileId < last_tile){
    int start = threadIdx.x + (tileId * TILE_SIZE);
    int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),sz );
        while(start < end){
          int id = start;
          #ifdef DEBUG
              assert(current_unique_nodes >= 0);
              assert(_tv1[id] < mask_size);
          #endif
          mask[_tv1[id]] = current_unique_nodes + id + 1;
          start += BLOCK_SIZE;
        }
    tileId += gridDim.x;
  }
}


template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void clear_mask(NDTYPE * mask, size_t mask_size,\
      NDTYPE*used_nodes, size_t used_nodes_size){
  int tileId = blockIdx.x;
  int last_tile = ((used_nodes_size - 1) / TILE_SIZE + 1);
  while(tileId < last_tile){
  int start = threadIdx.x + (tileId * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),used_nodes_size );
  while(start < end){
        int id = start;
  #ifdef DEBUG
      assert(used_nodes[id] < mask_size);
  #endif
    mask[used_nodes[id]] = 0;
    start += BLOCK_SIZE;
  }
    tileId += gridDim.x;
  }
}


//  Order based on blocks
template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void update_nodes(NDTYPE * mask,NDTYPE mask_size, NDTYPE* nodes, size_t node_size){
  int tileId = blockIdx.x;
  int last_tile = ((node_size - 1) / TILE_SIZE + 1);
  while(tileId < last_tile){
  int start = threadIdx.x + (tileId * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),node_size );
  while(start < end){
        int id = start;
    #ifdef DEBUG
        assert(nodes[id] < mask_size);
        if(mask[nodes[id]] == 0){
          printf("Not found %ld \n", nodes[id]);
        }
        assert(mask[nodes[id]] != 0);
    #endif
    nodes[id] = mask[nodes[id]] - 1;
     start += BLOCK_SIZE;
    }
    tileId += gridDim.x;
  }
}


ArrayMap::ArrayMap(NDTYPE num_nodes){
    gpuErrchk(cudaMalloc((void**)&mask, sizeof(NDTYPE) * num_nodes));
    gpuErrchk(cudaMemset(mask, 0, sizeof(NDTYPE) * num_nodes));
    mask_size = num_nodes;
    this->used_nodes.clear();
}

// Function changes the elements of nodes
void ArrayMap::remove_nodes_seen(device_vector<NDTYPE> &nodes){
  if(nodes.size() == 0)return;
  assert_no_duplicates(nodes);
  _tv.resize(nodes.size());
  _tv1.resize(nodes.size());
  int i = nodes.size();
  set_nodes_not_present<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(nodes.size()), BLOCK_SIZE>>>\
          (nodes.ptr(), nodes.size(),\
          mask, mask_size, (_tv.ptr()));

  // cudaDeviceSynchronize();
  NDTYPE nodes_not_seen = _tv[_tv.size()-1];
  _tv1 = _tv;


  transform<NDTYPE>::exclusive_scan(_tv, _tv1 );
  nodes_not_seen += _tv1[_tv1.size()-1];
  if(nodes_not_seen == 0){
    _tv1.clear();
    nodes.clear();
    _tv.clear();
    return;
  }

  _tv = _tv1;
  _tv1.resize(nodes_not_seen);
  // Capture all nodes not present
  // Step 2
  get_unique_nodes<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(nodes.size()), BLOCK_SIZE>>>\
    (nodes.ptr(), nodes.size(), \
      mask, mask_size, \
		 	_tv.ptr(),\
      (_tv1.ptr()), _tv1.size());
  // cudaDeviceSynchronize();    
  nodes = _tv1;
  _tv1.clear();
  _tv.clear();
}


void ArrayMap::assert_no_duplicates(device_vector<NDTYPE> &nodes){
  #ifdef DEBUG
      // check no duplicates;
      transform<NDTYPE>::unique(nodes, _tv);
      if(_tv.size()!= nodes.size()){
        _tv.debug("Unique");
        nodes.debug("Not Unique");
      }
      assert(_tv.size()  == nodes.size());
  #endif
}
// nodes has no duplicates
void ArrayMap::order(device_vector<NDTYPE> &nodes){
  if(nodes.size() == 0)return;
  #ifdef DEBUG
    assert_no_duplicates(nodes);  
  #endif
  _tv2 = nodes;

  remove_nodes_seen(_tv2);
  
  if(_tv2.size()== 0)return;
  // sort and get unique nodes
  // Step 3
  int current_unique_nodes = this->used_nodes.size();
  
  update_mask_with_unique<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(_tv2.size()), BLOCK_SIZE>>>\
      (mask, mask_size , current_unique_nodes,\
      (_tv2.ptr()),\
          _tv2.size());
  // Step 5
  this->used_nodes.append(_tv2);

  _tv2.clear();

}


void ArrayMap::clear(){
  if(this->used_nodes.size() == 0)return;
  clear_mask<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(used_nodes.size()), BLOCK_SIZE >>>\
    (mask, mask_size, \
      used_nodes.ptr(), used_nodes.size());
  this->used_nodes.clear();
 
}



void ArrayMap::replace(device_vector<NDTYPE> &nodes){
  
  if(nodes.size() == 0){ return;}
  update_nodes<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(nodes.size()), BLOCK_SIZE>>>\
      (mask, mask_size, (nodes.ptr()), nodes.size());

}

ArrayMap::~ArrayMap(){
    gpuErrchk(cudaFree(mask));
    this->used_nodes.clear();
}

device_vector<NDTYPE>& ArrayMap::get_used_nodes(){
  return this->used_nodes;
}
