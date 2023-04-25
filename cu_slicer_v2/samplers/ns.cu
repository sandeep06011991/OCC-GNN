#include "samplers.h"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "nvtx3/nvToolsExt.h"
#include "../util/cub.h"

__global__ void init_random_states(curandState *states, size_t num,
                                   unsigned long seed) {
  size_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
  assert(num == blockDim.x * gridDim.x);
  // Todo Add shared memory here
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
    cudaMalloc(&dev_curand_states, MAX_BLOCKS * BLOCK_SIZE * sizeof(curandState));
    init_random_states<<<MAX_BLOCKS, BLOCK_SIZE>>>(dev_curand_states,MAX_BLOCKS * BLOCK_SIZE , seed);
}

template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void sample_offsets(NDTYPE *in, size_t in_size, \
    NDTYPE *offsets_s, NDTYPE* in_degrees,\
     NDTYPE  *indptr_g, NDTYPE num_nodes, \
    int fanout, bool self_edge){
      int tileId = blockIdx.x;
      int last_tile = ((in_size - 1) / TILE_SIZE + 1);
      while(tileId < last_tile){
      int start = threadIdx.x + (tileId * TILE_SIZE);
      int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), in_size);

        while(start < end){
          int id = start;
          NDTYPE nd = in[id];
        #ifdef DEBUG
          assert(nd < num_nodes);
        #endif
        NDTYPE nbs_size = indptr_g[nd+1] - indptr_g[nd];
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
        start += BLOCK_SIZE;
        }
      tileId += gridDim.x;
    }
}

template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void neigh_sample_based_on_offsets(NDTYPE * in, NDTYPE size,\
    NDTYPE * offsets, NDTYPE * indices,\
      NDTYPE * graph_indptr, NDTYPE * graph_indices, NDTYPE num_nodes,\
         curandState *random_states, size_t num_random_states, int fanout,\
       bool self_edge){
     // Colascing random loads
     // Credit: Abhinav Jangda from nextdoor paper
    __shared__ unsigned char shRand[BLOCK_SIZE * sizeof(curandState)];

    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shRand[0];

    int* randStatesAsInts = (int*)random_states;

    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    auto curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
    int tileId = blockIdx.x;
    int last_tile = ((size - 1) / TILE_SIZE + 1);
    while(tileId < last_tile){
      int start = threadIdx.x + (tileId * TILE_SIZE);
      int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), size);
      while(start < end){
        int id = start;
        NDTYPE nd = in[id];
        #ifdef DEBUG
          assert(nd < num_nodes);
                // printf("%ld %ld %ld %ld\n", nd, offsets[nd], indices[offsets[id]], size);
                // assert(indices[offsets[nd]] < size);
        #endif
            // Todo
        NDTYPE nbs_size = graph_indptr[nd+1] -graph_indptr[nd];
        NDTYPE *read = &graph_indices[graph_indptr[nd]];

        NDTYPE *write = &indices[offsets[id]];
        if((nbs_size > fanout) && (fanout != -1)){
           for(int j = 0; j < fanout; j++){
              float f = curand_uniform(curandSrcPtr ) ;
              int sid = (int) (f * nbs_size - 1);
               #ifdef DEBUG
                if(sid >= nbs_size){
                    printf("overflow %f %d %ld\n", f, sid, nbs_size);
                  }
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
          start += BLOCK_SIZE;
        }
        tileId += gridDim.x;
      }
  }

// Not Reviewed
void NeighbourSampler::layer_sample(device_vector<NDTYPE> &in,
    device_vector<NDTYPE>  &in_degrees,
      device_vector<NDTYPE>  &offsets,
        device_vector<NDTYPE>  &indices, int fanout){
	//Fix me:
	//Do on gpu sampling
      offsets.clear();
      indices.clear();
      in_degrees.clear();
      offsets.resize(in.size() + 1);
      offsets.set_value(0,0);
      in_degrees.resize(in.size());
      sample_offsets<BLOCK_SIZE,TILE_SIZE><<<GRID_SIZE(in.size()), BLOCK_SIZE>>>
        (in.ptr(), in.size(), offsets.ptr(),\
          in_degrees.ptr(), this->dataset->indptr_d.ptr(), this->dataset->num_nodes, fanout, self_edge);
      gpuErrchk(cudaDeviceSynchronize());
      cuslicer::transform<NDTYPE>::inclusive_scan(offsets,offsets);
      gpuErrchk(cudaDeviceSynchronize());
      auto new_size = offsets[offsets.size() - 1];
      // thrust::inclusive_scan(thrust::device, offsets.begin(),\
      //      offsets.end(), offsets.begin()); // in-place scan
      indices.resize(new_size);
      neigh_sample_based_on_offsets<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(in.size()), BLOCK_SIZE>>>
      (in.ptr(), in.size(), \
            offsets.ptr(), \
             indices.ptr(), \
              this->dataset->indptr_d.ptr(), this->dataset->indices_d.ptr(), this->dataset->num_nodes,\
                  dev_curand_states, TOTAL_RAND_STATES, fanout, this->self_edge);
        gpuErrchk(cudaDeviceSynchronize());
}

void NeighbourSampler::sample(device_vector<NDTYPE> &target_nodes, Sample &s){
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
    cuslicer::transform<NDTYPE>::remove_duplicates(s.block[i]->indices,_t);
    dr->order(_t);
    // This line causes ptr copy and double destruction.
    // TODO: add a test for this and use shared ptr inside device vector
    device_vector<NDTYPE> us =   dr->get_used_nodes();
    s.block[i]->layer_nds.append(dr->get_used_nodes());
    dr->replace(s.block[i]->indices);
    gpuErrchk(cudaDeviceSynchronize());
  }
  // dr->clear();
  nvtxRangePop();
}
