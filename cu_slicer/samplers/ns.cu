#include "samplers/samplers.h"
#include <iostream>

__global__
void sample_offsets(long *in, size_t size, long *offsets,
    long* in_degrees, long  *indptr){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < size){
      long nd = in[id];
      long nbs_size = indptr[nd+1] - indptr[nd];
      in_degrees[id] = nbs_size;
      offsets[id+1] = nbs_size;
      id = id + (blockDim.x * gridDim.x);
  }
}

__global__
void neigh_sample_based_on_offsets(long * in, long size,
    long * offsets, long * indices,
      long * graph_indptr, long * graph_indices){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
      while(id < size){
          long nd = in[id];
          long sampled_nbs_size = offsets[id+1] - offsets[id];
          long *read = &graph_indices[graph_indptr[nd]];
          long *write = &indices[offsets[id]];
          for(int i = 0;i < sampled_nbs_size;i++){
            // Full neighbour sampling;
            write[i] = read[i];
          }
          id = id + (blockDim.x * gridDim.x);
      }
  }

void NeighbourSampler::layer_sample(thrust::device_vector<long> &in,
   thrust::device_vector<long> &in_degrees, thrust::device_vector<long> &offsets,
    thrust::device_vector<long> &indices){
	//Fix me:
	//Do on gpu sampling
      offsets.clear();
      indices.clear();
      in_degrees.clear();
      std::cout << "Offsets Size" << offsets.capacity() <<" \n";
      offsets.resize(10000);
      offsets.resize(in.size() + 1);
      // Arrays Work !!
      std::cout << "Capcityu Size" << offsets.capacity() <<" \n";
      std::cout << "Offsets Size" << offsets.size() <<" \n";

      offsets[0] =indices.size();
      in_degrees.resize(in.size());
      int blocks = (in.size()-1)/32 + 1;
      int threads = 32;
      sample_offsets<<<blocks, threads>>>
        (thrust::raw_pointer_cast(in.data()), in.size(), \
    thrust::raw_pointer_cast(offsets.data()),\
      thrust::raw_pointer_cast(in_degrees.data()),
        this->dataset->indptr);
       assert(in.size() < 640000);
       // Further optimizations are possible.
       // For lower number of neighbours

       thrust::inclusive_scan(thrust::device, offsets.begin(),\
            offsets.end(), offsets.begin()); // in-place scan


       indices.resize(offsets[offsets.size()-1]);
       neigh_sample_based_on_offsets<<<blocks, threads>>>
       (thrust::raw_pointer_cast(in.data()), in.size(),
            thrust::raw_pointer_cast(offsets.data()),
             thrust::raw_pointer_cast(indices.data()),
              this->dataset->indptr,
              this->dataset->indices);

  }

void NeighbourSampler::sample(thrust::device_vector<long> &target_nodes, Sample &s){
  s.block[0]->clear();
  s.block[0]->layer_nds = target_nodes;
  for(int i=1;i<s.num_layers+1;i++){
    s.block[i]->clear();
    std::cout << "Sample one layer\n";
    layer_sample(s.block[i-1]->layer_nds,s.block[i]->in_degree,
            s.block[i]->offsets, s.block[i]->indices);
    std::cout << "layer sampling complete\n";
    // Hack. Nodes for next layer must always be first.
    s.block[i]->layer_nds = s.block[i-1]->layer_nds;
    s.block[i]->layer_nds.insert(s.block[i]->layer_nds.end(), s.block[i]->indices.begin(), \
                s.block[i]->indices.end());

    dr->order_and_remove_duplicates(s.block[i]->layer_nds);
    std::cout << "Duplicate remover complete\n";
    dr->replace(s.block[i]->indices);
    dr->clear();
  }
}
