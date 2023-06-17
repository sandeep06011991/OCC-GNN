#include "slice.h"
#include <cstring>
#include "nvtx3/nvToolsExt.h"
#include "../util/cub.h"
#include "../util/array_utils.h"

using namespace cuslicer;


template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void calculate_cache_hit_mask(NDTYPE * in_nodes, OrderBook * orderbook,\
       size_t size, NDTYPE * cache_hit_mask, NDTYPE * cache_miss_mask, int partitionId){
  int tileId = blockIdx.x;
  int last_tile = ((size - 1) / TILE_SIZE + 1);
  while(tileId < last_tile){
  int start = threadIdx.x + (tileId * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), size);
  while(start < end){
      int tid = start; 
      long nd = in_nodes[tid];
      if(!orderbook->gpuContains(partitionId, nd)){
        cache_hit_mask[tid] = 0;
        cache_miss_mask[tid] = 1;
      }else{
        
	cache_hit_mask[tid] = 1;
        cache_miss_mask[tid] = 0;
      }
      start = start + BLOCK_SIZE;
    }
    tileId += gridDim.x;
  }
}

// Dont change this due to load balancing. 
template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void  fill_cache_nodes(NDTYPE * in_nodes,OrderBook * orderbook, size_t size,\
       NDTYPE * cache_hit_mask, NDTYPE * cache_miss_mask, \
			  NDTYPE * miss_from , NDTYPE * miss_to, NDTYPE  * hit_from, NDTYPE *hit_to, int partitionId){
        int tileId = blockIdx.x;
        int last_tile = ((size - 1) / TILE_SIZE + 1);
        while(tileId < last_tile){
        int start = threadIdx.x + (tileId * TILE_SIZE);
        int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), size);
  while(start < end){
        int tid = start;
        long nd = in_nodes[tid];
        if(!orderbook->gpuContains(partitionId, nd)){
                miss_from[cache_miss_mask[tid]-1] = nd;
		            miss_to[cache_miss_mask[tid]-1] = tid;
        }else{
                hit_from[cache_hit_mask[tid]-1] = orderbook->getLocalId(partitionId, nd);
                hit_to[cache_hit_mask[tid]-1] = tid;
        }
        start = start + BLOCK_SIZE;;

  }
    tileId += gridDim.x;
  }

}

void Slice::reorder(PartitionedLayer &l){\

    //   // Handle remote destination nodes
     for(int to = 0; to < this->num_gpus; to ++){
       // l.bipartite[to]->reorder_local(dr);
       dr->clear();
       dr->order(l.bipartite[to]->out_nodes_local);


       for(int from = 0; from < this->num_gpus; from++){
	       if(from == to) continue;
         dr->replace(l.bipartite[to]->push_from_ids[from]);
       }
     }

    for(int pull_from = 0;pull_from < this->num_gpus; pull_from++){
      dr->clear();
      dr->order(l.bipartite[pull_from]->in_nodes_local);
      for(int pull_to = 0; pull_to < this->num_gpus; pull_to ++ ){
        if(pull_from == pull_to)continue;
        dr->replace(l.bipartite[pull_from]->pull_to_ids[pull_to]);
    //     int start = l.bipartite[pull_to]->pull_from_offsets[pull_from];
    //     int end = l.bipartite[pull_to]->pull_from_offsets[pull_from + 1];
    //     thrust::device_vector<long> &f = l.bipartite[pull_from]->pull_to_ids[pull_to];
    //     thrust::device_vector<long> &t = l.bipartite[pull_to]->pulled_in_nodes;
    //     assert((end-start) <= t.size());
    //     f.clear();
    //     f.insert(f.end(), t.begin() + start, t.begin() + end);
    //     dr->replace(f);
      }
    }
    // nvtxRangePop();
  }



  void Slice::fill_cache_hits_and_misses(PartitionedSample &ps, int gpuid, device_vector<NDTYPE> &in_nodes){
  	cache_hit_mask.clear();
  	cache_miss_mask.clear();
  	cache_hit_mask.resize(in_nodes.size());
  	cache_miss_mask.resize(in_nodes.size());
    calculate_cache_hit_mask<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(in_nodes.size()), BLOCK_SIZE >>>(in_nodes.ptr(),\
  		       orderbook->getDevicePtr(),\
  			in_nodes.size(),\
  			cache_hit_mask.ptr(),\
  		  cache_miss_mask.ptr(),  gpuid);
    gpuErrchk(cudaDeviceSynchronize());

    cuslicer::transform<NDTYPE>::self_inclusive_scan(cache_hit_mask);

    cuslicer::transform<NDTYPE>::self_inclusive_scan(cache_miss_mask);

    //  thrust::inclusive_scan(cache_hit_mask.begin(), cache_hit_mask.end(), cache_hit_mask.begin());
  	//  thrust::inclusive_scan(cache_miss_mask.begin(), cache_miss_mask.end(), cache_miss_mask.begin());
  	 int misses = cache_miss_mask[in_nodes.size()-1];
  	 int hits = cache_hit_mask[in_nodes.size() - 1];

  	 ps.cache_miss_from[gpuid].resize(misses);
     ps.cache_hit_from[gpuid].resize(hits);
     ps.cache_miss_to[gpuid].resize(misses);
     ps.cache_hit_to[gpuid].resize(hits);
     assert(hits + misses == in_nodes.size());
  	 fill_cache_nodes<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(in_nodes.size()), BLOCK_SIZE>>>(in_nodes.ptr(),\
   		       orderbook->getDevicePtr(),\
   			in_nodes.size(),\
   			cache_hit_mask.ptr(),\
   		  cache_miss_mask.ptr(),\
        ps.cache_miss_from[gpuid].ptr(), ps.cache_miss_to[gpuid].ptr(),\
        ps.cache_hit_from[gpuid].ptr(), ps.cache_hit_to[gpuid].ptr(), gpuid);
      gpuErrchk(cudaDeviceSynchronize());

  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps, bool loadbalancing){
    // Get local partitioning Map 
    // Todo: 
    // 1. Partition last layer of sample nodes into local partition ids. 
    auto nodes = s.block[s.num_layers].layer_nds;
    this->sample_workload_map.resize(nodes.size());
    cuslicer::index_in<NDTYPE, PARTITIONIDX>(nodes, this->orderbook->getDevicePtr(), this->sample_workload_map);
    // this->workload_map
    // nodes.debug("In nodes");
    
    // this->sample_workload_map.debug("sample workload map");
    if(loadbalancing){
      this->loadbalancer->balance(nodes, this->sample_workload_map, \
            s.block[s.num_layers].layer_nds.size());
    }
    
    // Get partitioned layers.
    for(int i= 1; i< s.num_layers + 1;i++){
        bool last_layer = false;
        if (i == s.num_layers) last_layer = true;
    	  PartitionedLayer& l = ps.layers[i-1];
        this->slice_layer(s.block[i-1].layer_nds, \
            s.block[i], l, last_layer);
        // TODO: dont have to do this seperately.
        this->reorder(l);
        //consistency check
        #ifdef DEBUG
        for(int j = 0; j< this->num_gpus; j++){
          for(int k = 0; k < this->num_gpus;k ++ ){
            auto to = l.bipartite[j]->to_offsets[k + 1]  - l.bipartite[j]->to_offsets[k];
            assert(to == l.bipartite[k]->push_from_ids[j].size());
          }
        }
        #endif 
      }
      #ifdef DEBUG
        gpuErrchk(cudaDeviceSynchronize());
      #endif
      // Not can further optimize this.
       for(int i=0;i<this->num_gpus;i++){
           ps.cache_miss_from[i].clear();
           ps.cache_hit_from[i].clear();
           ps.cache_miss_to[i].clear();
           ps.cache_hit_to[i].clear();
           ps.last_layer_nodes[i].clear();
           auto in_nodes = ps.layers[s.num_layers- 1].bipartite[i]->in_nodes_local;
           if(in_nodes.size() > 0){
              fill_cache_hits_and_misses(ps, i, in_nodes);
          }
          ps.last_layer_nodes[i] = ps.layers[0].bipartite[i]->out_nodes_local;
      }
  
  }
