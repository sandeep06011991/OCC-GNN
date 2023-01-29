#include "transform/slice.h"
#include <cstring>
#include "graph/bipartite.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>


__global__
void partition_edges_push(int*  partition_map,\
    long * out_nodes, size_t out_nodes_size,\ // Sample layer out nodes indexed into the graph
      long *in_nodes, size_t in_nodes_size, \
        long * indptr, long *indices, long num_nodes_in_graph, \
        void ** indptr_map, void ** indices_map,\
      // Partitioned graphs such that indptr_map[dest, src]
       		bool last_layer, void ** storage_map, int NUM_GPUS){
            // Last layer use storage map
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < out_nodes_size){
      long nd1 = out_nodes[tid];
      long nbs = indptr[tid+1] - indptr[tid];
      #ifdef DEBUG
          assert(nd1 < num_nodes_in_graph);
      #endif
      int p_nd1 = partition_map[nd1];
      long offset_edge_start = indptr[tid];
      int p_nbs[MAX_DEVICES];
      for(int n=0; n<NUM_GPUS; n++){
        p_nbs[n] = 0;
      }
      p_nbs[p_nd1] = 1;
      for(int nb_idx = 0; nb_idx < nbs; nb_idx ++ ){
        long nd2_idx = indices[offset_edge_start + nb_idx];
        #ifdef DEBUG
            assert(nd2_idx < in_nodes_size);
        #endif
        long nd2 = in_nodes[nd2_idx];
        int p_nd2 = partition_map[nd2];
      	if(last_layer){
      		if(((int *)storage_map[p_nd1])[nd2]!= -1){
      			       p_nbs[p_nd1] = 1;
      	       		((long *)indices_map[p_nd1 * NUM_GPUS + p_nd1])\
                      [offset_edge_start + nd2_idx] = 1;
      			 continue;
      		}
      	}
	       p_nbs[p_nd2] = 1;
          // Denotes edge is selected
         ((long *)indices_map[p_nd1 * NUM_GPUS + p_nd2])[offset_edge_start + nb_idx] = 1;
      }

      for(int p_nd = 0; p_nd< NUM_GPUS ;p_nd++){
        if(p_nbs[p_nd] == 1){
          ((long *) indptr_map[p_nd * NUM_GPUS + p_nd])[tid] = 1;
        }
      }
      tid += (blockDim.x * gridDim.x);
    }
}

__global__
void partition_edges_pull(int*  partition_map,\
    long * out_nodes, size_t out_nodes_size,\ // Sample layer out nodes indexed into the graph
      long *in_nodes, size_t in_nodes_size, \
        long * indptr, long *indices, long num_nodes_in_graph, \
        void ** indptr_map, void ** indices_map,\
      // Partitioned graphs such that indptr_map[dest, src]
       		bool last_layer, void ** storage_map, int NUM_GPUS){
            // Last layer use storage map
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < out_nodes_size){
      long nd1 = out_nodes[tid];
      long nbs = indptr[tid+1] - indptr[tid];
      #ifdef DEBUG
          assert(nd1 < num_nodes_in_graph);
      #endif
      int p_nd1 = partition_map[nd1];
      long offset_edge_start = indptr[tid];

      for(int nb_idx = 0; nb_idx < nbs; nb_idx ++ ){
        long nd2_idx = indices[offset_edge_start + nb_idx];
        #ifdef DEBUG
            assert(nd2_idx < in_nodes_size);
        #endif
        long nd2 = in_nodes[nd2_idx];
        int p_nd2 = partition_map[nd2];
      	if(last_layer){
      		if(((int *)storage_map[p_nd1])[nd2]!= -1){
      			       p_nbs[p_nd1] = 1;
      	       		((long *)indices_map[p_nd1 * NUM_GPUS + p_nd1])\
                      [offset_edge_start + nd2_idx] = 1;
      			 continue;
      		}
      	}
          // In pull optimization always select edge
         ((long *)indices_map[p_nd1 * NUM_GPUS + p_nd1])[offset_edge_start + nb_idx] = 1;
         if(p_nd1 != p_nd2){
           ((long *)indices_map[p_nd1 * NUM_GPUS + p_nd2])[offset_edge_start + nb_idx] = 1;
         }
       }
        ((long *) indptr_map[p_nd1 * NUM_GPUS + p_nd1])[tid] = 1;
      }
      tid += (blockDim.x * gridDim.x);
    }
}

__global__ void populate_local_graphs_push(int*  partition_map, long * out_nodes,
      long *in_nodes, long * indptr, long *indices,\
        void ** indptr_index_map, void ** indices_index_map,
         void ** indptr_map, void ** indices_map,
            void ** to_nds_map, void ** out_degree_map,  int size,\
	    	bool last_layer, void ** storage_map){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size){
      long nd1 = out_nodes[tid];
      long nbs = indptr[tid+1] - indptr[tid];
      int p_nd1 = partition_map[nd1];
      long offset_edge_start = indptr[tid];
      int p_nbs[4];
      for(int i = 0; i<4; i++){
        p_nbs[i] = 0;
      }
      for(int n = 0; n<nbs; n ++ ){
        long nd2 = in_nodes[indices[offset_edge_start + n]];
        int p_nd2 = partition_map[nd2];
         if(last_layer){
                if(((int *)storage_map[p_nd1])[nd2]!= -1){
                        p_nbs[p_nd1] ++;
                        ((long *)indices_map[p_nd1 * 4 + p_nd1])[offset_edge_start + n] = nd2;
                         continue;
                }
        }

	     for(int i=0; i < 4; i++){
          if(i == p_nd2){
            p_nbs[i] ++;
              // ((long *) indptr_map[p_nd1 * 4 + p_nd2])[tid] = 1;
            // Denotes node is selected
            // Denotes edge is selected
            ((long *)indices_map[p_nd1 * 4 + p_nd2])\
                [((long *)indices_index_map[p_nd1 * 4 + p_nd2])[offset_edge_start + n]] \
                = nd2;
            }
        }
      }
      if(nbs == 0) nbs = 1;
      for(int p_nd2 = 0; p_nd2 < 4; p_nd2++){
        if(p_nbs[p_nd2] != 0){
          if(p_nd2 == p_nd1){
            ((long *)out_degree_map[p_nd1])\
                [((long *)indptr_index_map[p_nd1 * 4 + p_nd2])[tid] - 1] = nbs;
          }
        ((long *)indptr_map[p_nd1 * 4 + p_nd2])\
          [((long *)indptr_index_map[p_nd1 * 4 + p_nd2])[tid]] = p_nbs[p_nd2];
        ((long *)to_nds_map[p_nd1 * 4 + p_nd2])\
            [((long *)indptr_index_map[p_nd1 * 4 + p_nd2])[tid] - 1] = nd1;
        }
      }
      tid += (blockDim.x * gridDim.x);
    }
}

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


void Slice::slice_layer(thrust::device_vector<long> &layer_nds,
      Block &bs, PartitionedLayer &ps, bool last_layer){

    // Stage 1 Edge Partitioning
    ps.resize_index_and_offset_map(layer_nds.size(), bs.indices.size());
    if(pull_optimization){
      partition_edges_pull<<<BLOCK_SIZE(layer_nds.size()),THREAD_SIZE>>>\
        (thrust::raw_pointer_cast(this->workload_map.data()),\
          thrust::raw_pointer_cast(layer_nds.data()), layer_nds.size(),\
          thrust::raw_pointer_cast(bs.layer_nds.data()), bs.layer_nds.size(),\
          thrust::raw_pointer_cast(bs.offsets.data()),\
          thrust::raw_pointer_cast(bs.indices.data()),\
          this->num_nodes,\
            (void **)ps.device_offset_map, (void **)ps.device_indices_map,\
          last_layer, this->storage_map_flattened,this->num_gpus);
    }else{
      partition_edges_push<<<BLOCK_SIZE(layer_nds.size()),THREAD_SIZE>>>\
        (thrust::raw_pointer_cast(this->workload_map.data()),\
          thrust::raw_pointer_cast(layer_nds.data()), layer_nds.size(),\
          thrust::raw_pointer_cast(bs.layer_nds.data()), bs.layer_nds.size(),\
          thrust::raw_pointer_cast(bs.offsets.data()),\
          thrust::raw_pointer_cast(bs.indices.data()),\
          this->num_nodes,\
            (void **)ps.device_offset_map, (void **)ps.device_indices_map,\
          last_layer, this->storage_map_flattened,this->num_gpus);
    }

    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif

    // Stage 2 get sizes of Offsets for all graphs
    // Inclusive Scan
    int N = this->num_gpus * this->num_gpus;
    long local_graph_nodes[N];
    long local_graph_edges[N];
    for(int i=0; i< this->N; i++){
      thrust::inclusive_scan(ps.index_offset_map[i].begin(), ps.index_offset_map[i].end(), ps.index_offset_map[i].begin());
      thrust::inclusive_scan(ps.index_indices_map[i].begin(), ps.index_indices_map[i].end(), ps.index_indices_map[i].begin());
      local_graph_nodes[i] = ps.index_offset_map[i][ps.index_offset_map[i].size()-1];
      local_graph_edges[i] = ps.index_indices_map[i][ps.index_indices_map[i].size() - 1];
    }
    ps.resize_local_graphs(local_graph_nodes, local_graph_edges);
    // Stage 3 Populate local and remote edges.
    std::cout << "local graph partitioning \n";
    return;
    populate_local_graphs<<<1, 32>>>(thrust::raw_pointer_cast(this->workload_map.data()),
        thrust::raw_pointer_cast(layer_nds.data()),
        thrust::raw_pointer_cast(bs.layer_nds.data()),
        thrust::raw_pointer_cast(bs.offsets.data()),
        thrust::raw_pointer_cast(bs.indices.data()),
        (void **)ps.device_offset_map,
        (void **)ps.device_indices_map,
        (void **)ps.device_local_indptr_map,
        (void **)ps.device_local_indices_map,
        (void **)ps.device_local_to_nds_map,
        (void **)ps.device_out_nodes_degree_map,
        layer_nds.size(),\
        last_layer, this->storage_map_flattened);
    cudaDeviceSynchronize();
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
