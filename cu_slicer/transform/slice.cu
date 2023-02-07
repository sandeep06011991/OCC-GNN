#include "transform/slice.h"
#include <cstring>
#include "nvtx3/nvToolsExt.h"

__global__
void calculate_cache_hit_mask(long * in_nodes, int * storage_map, int size, int * cache_hit_mask, int * cache_miss_mask){
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if(tid < size){
   	long nd = in_nodes[tid];
	if(storage_map[nd] == -1){
		cache_hit_mask[tid] = 0;
		cache_miss_mask[tid] = 1;
	}else{
		cache_hit_mask[tid] = 1;
		cache_miss_mask[tid] = 0;
	}
	tid = tid + (blockDim.x * gridDim.x);
   }
}

__global__
void  fill_cache_nodes(long * in_nodes, int * storage_map, int size, int * cache_hit_mask, int * cache_miss_mask, \
			long * miss_from , long* miss_to, long * hit_from, long *hit_to){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if(tid < size){
        long nd = in_nodes[tid];
        if(storage_map[nd] == -1){
                miss_from[cache_miss_mask[tid]-1] = nd;
		            miss_to[cache_miss_mask[tid]-1] = tid;
        }else{
                hit_from[cache_hit_mask[tid]-1] = storage_map[nd];
                hit_to[cache_hit_mask[tid]-1] = tid;
        }
        tid = tid + (blockDim.x * gridDim.x);
   }

}

void Slice::reorder(PartitionedLayer &l){\
      // return;
       float t1;
      for(int i=0;i < this->num_gpus; i++){
       l.bipartite[i]->reorder_local(dr);

      }
      // Handle remote destination nodes
     nvtxRangePush("Non-local reorder");
     for(int to = 0; to < this->num_gpus; to ++){
       dr->clear();
       // l.bipartite[to]->debug();
       // Refactor not Done
       dr->order(l.bipartite[to]->out_nodes_local);
       for(int from = 0; from < this->num_gpus; from++){
	       if(from == to) continue;
         int start = l.bipartite[from]->to_offsets[to];
         int end = l.bipartite[from]->to_offsets[to + 1];
         l.bipartite[to]->push_from_ids[from].clear();

         thrust::device_vector<long> &t = l.bipartite[to]->push_from_ids[from];
      	 thrust::device_vector<long> &f = l.bipartite[from]->out_nodes_remote;
         t.insert(t.end(), f.begin() + start, f.begin() + end );
      	 dr->replace(t);
       }
    }
    for(int pull_from = 0;pull_from < this->num_gpus; pull_from++){
      dr->clear();
      dr->order(l.bipartite[pull_from]->in_nodes);
      for(int pull_to = 0; pull_to < this->num_gpus; pull_to ++ ){
        if(pull_from == pull_to)continue;
        int start = l.bipartite[pull_to]->pull_from_offsets[pull_from];
        int end = l.bipartite[pull_to]->pull_from_offsets[pull_from + 1];
        thrust::device_vector<long> &f = l.bipartite[pull_from]->pull_to_ids[pull_to];
        thrust::device_vector<long> &t = l.bipartite[pull_to]->pulled_in_nodes;
        assert((end-start) <= t.size());
        f.clear();
        f.insert(f.end(), t.begin() + start, t.begin() + end);
        dr->replace(f);
      }
    }
    nvtxRangePop();
  }



  void Slice::fill_cache_hits_and_misses(PartitionedSample &ps, int gpuid, thrust::device_vector<long> &in_nodes){
  	cache_hit_mask.clear();
  	cache_miss_mask.clear();
  	cache_hit_mask.resize(in_nodes.size());
  	cache_miss_mask.resize(in_nodes.size());

  	 calculate_cache_hit_mask<<<BLOCK_SIZE(in_nodes.size()), THREAD_SIZE>>>(thrust::raw_pointer_cast(in_nodes.data()),\
  		       	thrust::raw_pointer_cast(storage_map[gpuid].data()),\
  			in_nodes.size(),\
  			thrust::raw_pointer_cast(cache_hit_mask.data()),\
  			thrust::raw_pointer_cast(cache_miss_mask.data()));
         gpuErrchk(cudaDeviceSynchronize());
  	 thrust::inclusive_scan(cache_hit_mask.begin(), cache_hit_mask.end(), cache_hit_mask.begin());
  	 thrust::inclusive_scan(cache_miss_mask.begin(), cache_miss_mask.end(), cache_miss_mask.begin());
  	 int misses = cache_miss_mask[in_nodes.size()-1];
  	 int hits = cache_hit_mask[in_nodes.size() - 1];
  	 ps.cache_miss_from[gpuid].resize(misses);
           ps.cache_hit_from[gpuid].resize(hits);
           ps.cache_miss_to[gpuid].resize(misses);
           ps.cache_hit_to[gpuid].resize(hits);

  	 fill_cache_nodes<<<BLOCK_SIZE(in_nodes.size()), THREAD_SIZE>>>(thrust::raw_pointer_cast(in_nodes.data()),\
                          thrust::raw_pointer_cast(storage_map[gpuid].data()),\
                          in_nodes.size(),
                          thrust::raw_pointer_cast(cache_hit_mask.data()),\
                          thrust::raw_pointer_cast(cache_miss_mask.data()),\
  	thrust::raw_pointer_cast(ps.cache_miss_from[gpuid].data()), thrust::raw_pointer_cast(ps.cache_miss_to[gpuid].data()),\
  	thrust::raw_pointer_cast(ps.cache_hit_from[gpuid].data()), thrust::raw_pointer_cast(ps.cache_hit_to[gpuid].data()));
     gpuErrchk(cudaDeviceSynchronize());


  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps){
    float _t1,_t2,slice_time, reorder, cache;
    slice_time = 0;
    reorder = 0;
    cache = 0;
     for(int i= 1; i< s.num_layers + 1;i++){
        bool last_layer = false;
        if (i == s.num_layers) last_layer = true;
    	  PartitionedLayer& l = ps.layers[i-1];

        nvtxRangePush("Slice");
        this->slice_layer(s.block[i-1]->layer_nds, \
            (* s.block[i]), l, last_layer);
        nvtxRangePop();
        this->reorder(l);

      }
      #ifdef DEBUG
        gpuErrchk(cudaDeviceSynchronize());
      #endif
	cudaEventRecord(event1,0);
       nvtxRangePush("cache");
       for(int i=0;i<this->num_gpus;i++){
           ps.cache_miss_from[i].clear();
           ps.cache_hit_from[i].clear();
           ps.cache_miss_to[i].clear();
           ps.cache_hit_to[i].clear();
           ps.last_layer_nodes[i].clear();
           thrust::device_vector<long> &in_nodes = ps.layers[s.num_layers- 1].bipartite[i]->in_nodes;
           if(in_nodes.size() > 0){
              fill_cache_hits_and_misses(ps, i, in_nodes);
          }
  	   thrust::device_vector<long> &last_layer = ps.layers[0].bipartite[i]->out_nodes_local;
       ps.last_layer_nodes[i].insert(ps.last_layer_nodes[i].end(), last_layer.begin(), last_layer.end());
      }
      nvtxRangePop();
       cudaEventRecord(event2,0);
 }
