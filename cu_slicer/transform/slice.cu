#include "transform/slice.h"
#include <cstring>


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
                miss_from[cache_miss_mask[tid]] = nd;
		miss_to[cache_miss_mask[tid]] = tid;
        }else{
                hit_from[cache_hit_mask[tid]] = storage_map[nd];
                hit_to[cache_hit_mask[tid]] = tid;
        }
        tid = tid + (blockDim.x * gridDim.x);
   }

}

void Slice::reorder(PartitionedLayer &l){
    // Not checked
    return;
    for(int i=0;i < this->num_gpus; i++){
      std::cout << "local reordering " << i <<"\n";
       l.bipartite[i]->reorder_local(dr);
     }
     // return;
     // Handle remote destination nodes
     for(int to = 0; to < this->num_gpus; to ++){
       dr->clear();
       // l.bipartite[to]->debug();
       std::cout << "OUT nodes local \n";
       for(auto o: l.bipartite[to]->out_nodes_local ){
         std::cout << o <<" ";
       }
       std::cout << "\n";
       remove_duplicates(l.bipartite[to]->out_nodes_local);
       dr->order(l.bipartite[to]->out_nodes_local);
       for(int from = 0; from < this->num_gpus; from++){
	        if(from == to) continue;
         int start = l.bipartite[from]->to_offsets[to];
         int end = l.bipartite[from]->to_offsets[to + 1];
         l.bipartite[to]->from_ids[from].clear();

         thrust::device_vector<long> &t = l.bipartite[to]->from_ids[from];
      	 thrust::device_vector<long> &f = l.bipartite[from]->out_nodes_remote;
         t.insert(t.end(), f.begin() + start, f.begin() + end );
         std::cout << "to" << start << ":" << end <<":"<< f.size() <<"\n";
      	 dr->replace(t);
       }
     }


  }



  void Slice::fill_cache_hits_and_misses(PartitionedSample &ps, int gpuid, thrust::device_vector<long> &in_nodes){
  	cache_hit_mask.clear();
  	cache_miss_mask.clear();
  	cache_hit_mask.resize(in_nodes.size());
  	cache_miss_mask.resize(in_nodes.size());
  	assert(in_nodes.size() < 40000);
  	int blocks = (in_nodes.size() -1)/32 + 1;
    std::cout << storage_map[gpuid].size() <<"crss " << in_nodes.size() <<"\n";
  	 calculate_cache_hit_mask<<<blocks, 32>>>(thrust::raw_pointer_cast(in_nodes.data()),\
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

  	 fill_cache_nodes<<<blocks, 32>>>(thrust::raw_pointer_cast(in_nodes.data()),\
                          thrust::raw_pointer_cast(storage_map[gpuid].data()),\
                          in_nodes.size(),
                          thrust::raw_pointer_cast(cache_hit_mask.data()),\
                          thrust::raw_pointer_cast(cache_miss_mask.data()),\
  	thrust::raw_pointer_cast(ps.cache_miss_from[gpuid].data()), thrust::raw_pointer_cast(ps.cache_miss_to[gpuid].data()),\
  	thrust::raw_pointer_cast(ps.cache_hit_from[gpuid].data()), thrust::raw_pointer_cast(ps.cache_hit_to[gpuid].data()));
     gpuErrchk(cudaDeviceSynchronize());


  }

  void Slice::slice_sample(Sample &s, PartitionedSample &ps){
      for(int i= 1; i< s.num_layers + 1;i++){
        bool last_layer = false;
        if (i == s.num_layers) last_layer = true;
    	  PartitionedLayer& l = ps.layers[i-1];
        std::cout << "Slicing starts" <<"\n";
        this->slice_layer(s.block[i-1]->layer_nds, \
            (* s.block[i]), l, last_layer);
          this->reorder(l);
      }
      std::cout << "Cache handlig \n";
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
  	 //       // ps.layers[s.num_layers-1].bipartite[i]->debug();
          // for(int j = 0; j <ps.layers[s.num_layers- 1].bipartite[i]->in_nodes.size(); j++){
            // auto nd = ps.layers[s.num_layers-1].bipartite[i]->in_nodes[j];
      //       if (this->storage_map[i][nd] != -1){
      //           ps.cache_hit_from[i].push_back(this->storage_map[i][nd]);
      //           ps.cache_hit_to[i].push_back(j);
      //       }else{
      //         ps.cache_miss_from[i].push_back(nd);
      //         ps.cache_miss_to[i].push_back(j);
      //       }
      //     }
  	thrust::device_vector<long> &last_layer = ps.layers[0].bipartite[i]->out_nodes_local;
          ps.last_layer_nodes[i].insert(ps.last_layer_nodes[i].end(), last_layer.begin(), last_layer.end());
      }
    }
