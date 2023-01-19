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
using namespace std;

__global__
void set_zero(int *array, int size){
    int id= blockDim.x * blockIdx.x + threadIdx.x;
    while(id <size){
    	 array[id] = 0;
	     id += gridDim.x * blockDim.x;
    }
}

__global__
void clear_mask(int *mask, long *nodes, int size){
  int id= blockDim.x * blockIdx.x + threadIdx.x;
  while(id <size){
    mask[nodes[id]] = 0;
    id += gridDim.x * blockDim.x;
  }
}

__global__
void set_nodes_not_present(long * nodes, int size, int * mask, long *_tv){
   int id= blockDim.x * blockIdx.x + threadIdx.x;
   //printf("%d ", size);
   while(id <size){
   //     printf("%d %ld\n",mask[nodes[id]],nodes[id]);
      if(mask[nodes[id]] == 0){
        _tv[id] = 1;
       }else{
	     _tv[id] = 0;
       }
       id += gridDim.x * blockDim.x;
    }
}

// Nodes not already present
__global__
void get_unique_nodes(long *nodes, int size, int *mask, long *_tv, long *_tv1){
   int id= blockDim.x * blockIdx.x + threadIdx.x;
    while(id <size){
        if(mask[nodes[id]] == 0){
           _tv1[_tv[id]] = nodes[id];
        }else{
          // _tv[id] = 0;
        }
        id += gridDim.x * blockDim.x;
    }
}


__global__
void update_mask_with_unique(int *mask, int current_unique_nodes, long * _tv1, int size){
    int id= blockDim.x * blockIdx.x + threadIdx.x;
    while(id <size){
	     // printf("updating %d %ld\n",id, _tv1[id]);
        mask[_tv1[id]] = current_unique_nodes + id + 1;
        id += gridDim.x * blockDim.x;
    }
}

__global__
void update_nodes(int *mask,long *_tv1, int size){
   int id= blockDim.x * blockIdx.x + threadIdx.x;
   while(id <size){
       _tv1[id] = mask[_tv1[id]]-1;
       // printf("update node %d %d\n", id, _tv1[id]);
       id += gridDim.x * blockDim.x;

   }

}


ArrayMap::ArrayMap(long num_nodes){
    gpuErrchk(cudaMalloc((void**)&mask, sizeof(int) * num_nodes));
    //mask = (int *)malloc(sizeof(int) * num_nodes);
    int num_blocks = ((num_nodes - 1)/64 + 1);
    int num_threads = 64;
    set_zero<<<num_blocks, num_threads>>>(mask,num_nodes);
    gpuErrchk( cudaDeviceSynchronize() );
    this->used_nodes.clear();
}



void ArrayMap::order_and_remove_duplicates(thrust::device_vector<long>& nodes){
  thrust::sort(nodes.begin(), nodes.end());
  thrust::unique(nodes.begin(), nodes.end());
  order(nodes);
 }

void debug_vector(thrust::device_vector<long>& out){
	thrust::host_vector<long> in = out;
	std::cout << "debug";
	for(auto a:in){
	  std::cout << a << " ";
	}
	std::cout <<"\n";
}

void ArrayMap::order(thrust::device_vector<long> &nodes){
  int i = nodes.size();
  _tv.resize(nodes.size());
  _tv1.resize(nodes.size());
  // Step 1
  int num_blocks = (i - 1)/64 + 1;
  int num_threads = 64;
  set_nodes_not_present<<<num_blocks, num_threads>>>(\
			  thrust::raw_pointer_cast(nodes.data()), i, mask, \
			  thrust::raw_pointer_cast(_tv.data()));
  gpuErrchk( cudaDeviceSynchronize() );

  thrust::exclusive_scan(thrust::device, _tv.begin() , _tv.end(), _tv.begin(), 0); // in-place scan
  _tv1.resize(_tv[_tv.size()-1] + 1);
  // Capture all nodes not present
  // Step 2
  get_unique_nodes<<<num_blocks,64>>>(thrust::raw_pointer_cast(nodes.data()), nodes.size(), mask,\
		 	thrust::raw_pointer_cast(_tv.data()),thrust::raw_pointer_cast(_tv1.data()));

  std::cout << "original" << _tv1.size() <<"\n";
  thrust::sort(_tv1.begin(), _tv1.end());
  _tv.clear();
  _tv.resize(_tv1.size());
  auto it = thrust::unique_copy(_tv1.begin(), _tv1.end(), _tv.begin());
  _tv.erase(it, _tv.end());
  // _tv1.resize(count);
  // std::cout << v <<"\n";
  std::cout << "Cleaned duplicates " << _tv.size() <<"\n";
  // sort and get unique nodes
  // Step 3
  int current_unique_nodes = this->used_nodes.size();
  num_blocks = (current_unique_nodes - 1)/64 + 1;
  gpuErrchk(cudaDeviceSynchronize());

  update_mask_with_unique<<<num_blocks, 64>>>( mask, current_unique_nodes,
		  thrust::raw_pointer_cast(_tv.data()), _tv.size());
  // Update Mask
  // Step 4
  // Add to used nodes
  //num_blocks = (i-1)/64 + 1;
  //update_nodes<<<num_blocks, 64>>>(mask, thrust::raw_pointer_cast(nodes.data()),\
		nodes.size());
  //gpuErrchk(cudaDeviceSynchronize());

  // Step 5
  this->used_nodes.insert(this->used_nodes.end(), _tv.begin(), _tv.end());
  _tv1.clear();
  _tv.clear();

}



void ArrayMap::clear(){
  int i = this->used_nodes.size();
  int num_blocks = (i-1)/64 + 1;
  clear_mask<<<num_blocks,64>>>(mask, thrust::raw_pointer_cast(this->used_nodes.data()), used_nodes.size());
  this->used_nodes.clear();
}


void ArrayMap::replace(thrust::device_vector<long> &nodes){
  int i = nodes.size();
  int num_blocks = (i-1)/64 + 1;
  update_nodes<<<num_blocks, 64>>>(mask, thrust::raw_pointer_cast(nodes.data()),\
                nodes.size());
  gpuErrchk(cudaDeviceSynchronize());
}

ArrayMap::~ArrayMap(){
    gpuErrchk(cudaFree(mask));
    this->used_nodes.clear();
}

int test_duplicate(){
    long src[] = { 1, 2, 3, 2, 3, 4, 5 };
    int n = sizeof(src) / sizeof(src[0]);
    cudaSetDevice(0);
    thrust::host_vector<long> h_vec(src, src + n);
    thrust::device_vector<long> d_vec = h_vec;
    ArrayMap * map = new ArrayMap(10);
    std::cout <<"Attempt reorder \n";
    map->order(d_vec);
    h_vec = d_vec;
    for(auto i :h_vec){
	    std::cout << i <<" ";
    }
    std::cout <<"\n";
    std::cout << "test 1 done move on\n";
    return 1;
}


/*
   Not extending this into the GPU.
HashMap::HashMap(long num_nodes){
    map.clear();
    count = 0;
}

void HashMap::clear(){
  map.clear();
  count = 0;
}

void HashMap::order_and_remove_duplicates(std::vector<long>& nodes){
    // assert(this->used_nodes.size() == 0);
    int j = 0;
    for(long nd1: nodes){
      auto search = map.find(nd1);
      if(search == map.end()){
        //fixme: Potential firehazard
        count++;
        map.insert(std::make_pair(nd1, count));
        nodes[j] = nd1;
        j++;
      }
    }
    nodes.resize(j);
}

void HashMap::order(std::vector<long> &nodes){
  for(long nd1: nodes){
    auto search = map.find(nd1);
    if(search == map.end()){
      //fixme: Potential firehazard
      count++;
      map.insert(std::make_pair(nd1, count));
    }
  }

}

void HashMap::replace(vector<long> &v){
  int failed = 0;
  for(int i=0;i<v.size();i++){
    long t = v[i];
    auto search = map.find(t);

    if(search == map.end()){
      std::cout << "failed to find " <<  t <<"\n";
      failed ++ ;
      assert(false);
    }
    v[i] = search->second -1;
  }
  // std::cout << "failed for " << failed << " " << v.size()  <<"\n";
}
*/
