#include "duplicate.h"
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <util/cuda_utils.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <kernels/parallel_for.cuh>
#include <thrust/fill.h>
using namespace std;

__global__
void set_nodes_not_present(long * nodes, size_t nodes_size,
        int * mask, size_t mask_size, long *_tv){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    while(id < nodes_size){
      #ifdef DEBUG
          assert(nodes[id] < mask_size);
      #endif
      if(mask[nodes[id]] == 0){
            _tv[id] = 1;
           }else{
           _tv[id] = 0;
       }
       id += (blockDim.x * gridDim.x);
    }
}

__global__
void get_unique_nodes(long *nodes, size_t nodes_size,
      int *mask, size_t mask_size,
        long *_tv, long *_tv1, size_t tv1_size){
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  while(id < nodes_size){
    #ifdef DEBUG
        assert(nodes[id] < mask_size);
    #endif
    if(mask[nodes[id]] == 0){
      #ifdef DEBUG
        assert(_tv[id] < tv1_size);
      #endif
     _tv1[_tv[id]] = nodes[id];
      }
      id += (blockDim.x * gridDim.x);
   }
}


__global__
void update_mask_with_unique(int *mask, long mask_size,
      int current_unique_nodes, long * _tv1, size_t sz){
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    while(id < sz){
      #ifdef DEBUG
          assert(current_unique_nodes >= 0);
          assert(_tv1[id] < mask_size);
      #endif
      mask[_tv1[id]] = current_unique_nodes + id + 1;
      id += (blockDim.x * gridDim.x);
    }
}



ArrayMap::ArrayMap(long num_nodes){
    gpuErrchk(cudaMalloc((void**)&mask, sizeof(int) * num_nodes));
    mask_size = num_nodes;
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(mask);
    thrust::fill(dev_ptr, dev_ptr + num_nodes, 00);
    this->used_nodes.clear();
}

// Function changes the elements of nodes
void ArrayMap::remove_nodes_seen(thrust::device_vector<long> &nodes){
  if(nodes.size() == 0)return;
  assert_no_duplicates(nodes);
  _tv.resize(nodes.size());
  _tv1.resize(nodes.size());
  int i = nodes.size();

  set_nodes_not_present<<<BLOCK_SIZE(nodes.size()), THREAD_SIZE>>>\
          (thrust::raw_pointer_cast(nodes.data()), nodes.size(),\
          mask, mask_size, thrust::raw_pointer_cast(_tv.data()));

  int nodes_not_seen = _tv[_tv.size()-1];
  thrust::exclusive_scan(thrust::device, _tv.begin() , _tv.end(), _tv.begin(), 0); // in-place scan
  nodes_not_seen += _tv[_tv.size()-1];
  if(nodes_not_seen == 0){
    _tv1.clear();
    nodes.clear();
    _tv.clear();
    return;
  }
  _tv1.resize(nodes_not_seen);
  // Capture all nodes not present
  // Step 2
  get_unique_nodes<<<BLOCK_SIZE(nodes.size()), THREAD_SIZE>>>\
    (thrust::raw_pointer_cast(nodes.data()), nodes.size(), \
      mask, mask_size, \
		 	thrust::raw_pointer_cast(_tv.data()),\
      thrust::raw_pointer_cast(_tv1.data()), _tv1.size());
  nodes = _tv1;
  _tv1.clear();
  _tv.clear();
}


void ArrayMap::assert_no_duplicates(thrust::device_vector<long> &nodes){
  #ifdef DEBUG
      // check no duplicates;
      _tv = nodes;
      remove_duplicates(_tv);
      assert(_tv.size()  == nodes.size());
  #endif
}
// nodes has no duplicates
void ArrayMap::order(thrust::device_vector<long> &nodes){
  if(nodes.size() == 0)return;
  _tv2 = nodes;
  remove_nodes_seen(_tv2);
  if(_tv2.size()== 0)return;
  // sort and get unique nodes
  // Step 3
  int current_unique_nodes = this->used_nodes.size();
  update_mask_with_unique<<<BLOCK_SIZE(_tv2.size()), THREAD_SIZE>>>\
      (mask, mask_size, current_unique_nodes,\
      thrust::raw_pointer_cast(_tv2.data()),\
          _tv2.size());
  // Step 5
  this->used_nodes.insert(this->used_nodes.end(), _tv2.begin(), _tv2.end());
  _tv2.clear();

}

__global__
void clear_mask(int * mask, long mask_size,\
      long *used_nodes, size_t used_nodes_size){
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  while(id < used_nodes_size){
    #ifdef DEBUG
        assert(used_nodes[id] < mask_size);
    #endif
      mask[used_nodes[id]] = 0;
      id += (blockDim.x * gridDim.x);
  }
}

void ArrayMap::clear(){
  if(this->used_nodes.size() == 0)return;
  clear_mask<<<BLOCK_SIZE(used_nodes.size()), THREAD_SIZE>>>\
    (mask, mask_size, \
      thrust::raw_pointer_cast(used_nodes.data()), used_nodes.size());
  this->used_nodes.clear();
}


__global__
void update_nodes(int * mask,long  mask_size, long * nodes, size_t node_size){
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  while(id <  node_size){
    #ifdef DEBUG
        assert(nodes[id] < mask_size);
        if(mask[nodes[id]] == 0){
          printf("Not found %ld \n", nodes[id]);
        }
        assert(mask[nodes[id]] != 0);
    #endif
     nodes[id] = mask[nodes[id]] - 1;
     id += (blockDim.x * gridDim.x);
  }
}

void ArrayMap::replace(thrust::device_vector<long> &nodes){
  update_nodes<<<BLOCK_SIZE(nodes.size()), THREAD_SIZE>>>\
      (mask, mask_size, thrust::raw_pointer_cast(nodes.data()), nodes.size());

}

ArrayMap::~ArrayMap(){
    gpuErrchk(cudaFree(mask));
    this->used_nodes.clear();
}

thrust::device_vector<long>& ArrayMap::get_used_nodes(){
  return this->used_nodes;
}

void test_duplicate(){
    long src[] = { 14, 12, 9, 9 };
    cudaSetDevice(0);
    thrust::host_vector<long> h_vec(std::begin(src), std::end(src));
    thrust::device_vector<long> d_vec = h_vec;
    ArrayMap * map = new ArrayMap(2000);
    remove_duplicates(d_vec);
    map->order(d_vec);
    map->replace(d_vec);
    long src1[] = {9, 15};
    thrust::host_vector<long> h_vec1(std::begin(src1), std::end(src1));
    thrust::device_vector<long> d_vec1 = h_vec1;
    map->order(d_vec1);
    map->replace(d_vec1);
    map->clear();
    d_vec1 = h_vec1;
    map->order(d_vec1);
    map->replace(d_vec1);
    h_vec = d_vec1;
    // for(auto i :h_vec){
	  //   std::cout << i <<" ";
    // }
    // std::cout <<"\n";
    map->clear();
    h_vec.clear();
    for(int i = 0; i <1024; i += 2){
      h_vec.push_back(i);
    }

    d_vec = h_vec;
    map->order(d_vec);
    map->replace(d_vec);
    h_vec.clear();
    for(int i=0;i < 512; i++){
      h_vec.push_back(i);
    }
    // std::cout <<"miss" << h_vec.size() << " " <<  d_vec1.size() << "\n";
    checkVectorSame<long>(h_vec, d_vec);
    std::cout << "test 1 done move on\n";

}
