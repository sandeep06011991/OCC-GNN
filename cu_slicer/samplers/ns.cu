#include "samplers/samplers.h"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include "nvtx3/nvToolsExt.h"

__global__ void init_random_states(curandState *states, size_t num,
                                   unsigned long seed) {
  size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  assert(num == blockDim.x * gridDim.x);
  if (threadId < num) {
    // Copied from GNNLAB
    /** Using different seed & constant sequence 0 can reduce memory
      * consumption by 800M
      * https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
      */
    curand_init(seed+threadId, 0, 0, &states[threadId]);
  }
}

NeighbourSampler::NeighbourSampler(std::shared_ptr<Dataset> dataset,
    vector<int> fanout, bool self_edge){
    this->dataset = dataset;
    this->fanout = fanout;
    dr = new ArrayMap(this->dataset->num_nodes);
    this->self_edge = self_edge ;
    unsigned long seed = \
      std::chrono::system_clock::now().time_since_epoch().count();
    cudaMalloc(&dev_curand_states, TOTAL_RAND_STATES * sizeof(curandState));
    init_random_states<<<MAX_BLOCKS, THREAD_SIZE>>>(dev_curand_states, TOTAL_RAND_STATES, seed);
}


__global__
void sample_offsets(long *in, size_t in_size, \
    long *offsets_s, long* in_degrees,\
     long  *indptr_g, long num_nodes, \
    int fanout, bool self_edge){
      int id = blockIdx.x * blockDim.x + threadIdx.x;
  while(id < in_size){
      long nd = in[id];
      #ifdef DEBUG
        assert(nd < num_nodes);
      #endif
      long nbs_size = indptr_g[nd+1] - indptr_g[nd];
      if(fanout != -1){
        if(fanout < nbs_size){
            nbs_size = fanout;
        }
      }
      if(nbs_size == 0){
      	in_degrees[id] = 1;
      }else{
        in_degrees[id] = nbs_size;
      }
      if(self_edge) nbs_size += 1;
      // TODO: Check impact of misaligned access
      offsets_s[id+1] = nbs_size;
      id = id + (blockDim.x * gridDim.x);
  }
}

__global__
void neigh_sample_based_on_offsets(long * in, long size,\
    long * offsets, long * indices,\
      long * graph_indptr, long * graph_indices, long num_nodes,\
         curandState *random_states, size_t num_random_states, int fanout,\
       bool self_edge){
      int threadId =  blockIdx.x * blockDim.x + threadIdx.x;
      int id = threadId;
      while(id < size){
          long nd = in[id];
          #ifdef DEBUG
              assert(nd < num_nodes);
              assert(indices[offsets[id]] < size);
          #endif
          // Todo
          long nbs_size = graph_indptr[nd+1] -graph_indptr[nd];
          long *read = &graph_indices[graph_indptr[nd]];

          long *write = &indices[offsets[id]];
          if((nbs_size > fanout) && (fanout != -1)){
             for(int j = 0; j < fanout; j++){
               int sid = (int) (curand_uniform(&random_states[threadId]) * nbs_size);
               #ifdef DEBUG
                    assert(sid < nbs_size);
               #endif
               write[j] = read[sid];
             }
             if(self_edge){
               write[fanout] = nd;
             }
          }else{
              for(int j = 0; j < nbs_size; j++){
                write[j] = read[j];
              }
              if(self_edge){
                write[nbs_size] = nd;
              }
          }
          id = id + (blockDim.x * gridDim.x);
      }
  }

// Not Reviewed
void NeighbourSampler::layer_sample(thrust::device_vector<long> &in,
    thrust::device_vector<long> &in_degrees,
      thrust::device_vector<long> &offsets,
        thrust::device_vector<long> &indices, int fanout){
	//Fix me:
	//Do on gpu sampling
      offsets.clear();
      indices.clear();
      in_degrees.clear();
      offsets.resize(in.size() + 1);
      offsets[0] = 0;
      in_degrees.resize(in.size());
      sample_offsets<<<BLOCK_SIZE(in.size()), THREAD_SIZE>>>
        (thrust::raw_pointer_cast(in.data()), in.size(), \
    thrust::raw_pointer_cast(offsets.data()),\
      thrust::raw_pointer_cast(in_degrees.data()),
        this->dataset->indptr, this->dataset->num_nodes, fanout, self_edge);

       thrust::inclusive_scan(thrust::device, offsets.begin(),\
            offsets.end(), offsets.begin()); // in-place scan


       indices.resize(offsets[offsets.size()-1]);

       neigh_sample_based_on_offsets<<<BLOCK_SIZE(in.size()), THREAD_SIZE>>>
       (thrust::raw_pointer_cast(in.data()), in.size(),
            thrust::raw_pointer_cast(offsets.data()),
             thrust::raw_pointer_cast(indices.data()),
              this->dataset->indptr, this->dataset->indices, this->dataset->num_nodes,\
                  dev_curand_states, TOTAL_RAND_STATES, fanout, this->self_edge);


  }

void NeighbourSampler::sample(thrust::device_vector<long> &target_nodes, Sample &s){
  nvtxRangePush("sample");
  s.block[0]->clear();
  dr->clear();
  dr->order(target_nodes);
  s.block[0]->layer_nds = target_nodes;
  assert(s.num_layers == this->fanout.size());
  for(int i=1;i<s.num_layers+1;i++){
    s.block[i]->clear();
    layer_sample(s.block[i-1]->layer_nds, s.block[i]->in_degree,
            s.block[i]->offsets,  s.block[i]->indices, this->fanout[i-1]);
    _t.clear();
    _t = s.block[i]->indices;
    remove_duplicates(_t);
    dr->order(_t);
    auto used = dr->get_used_nodes();
    s.block[i]->layer_nds.insert(s.block[i]->layer_nds.end(),
          used.begin(), used.end());
    dr->replace(s.block[i]->indices);
  }
  dr->clear();
  nvtxRangePop();
}
