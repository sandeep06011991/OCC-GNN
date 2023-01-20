#include "transform/slice.h"
#include <cstring>
#include "graph/bipartite.h"

// Keep this a bit more consistant
__global__ void partition_edges(int*  partition_map, long * out_nodes,
  long *in_nodes, long * indptr, long *indices,\
        void ** indptr_map, void ** indices_map, int size){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size){
      long nd1 = out_nodes[tid];
      long nbs = indptr[tid+1] - indptr[tid];
      int p_nd1 = partition_map[nd1];
      long offset_edge_start = indptr[tid];
      for(int n = 0; n<nbs; n ++ ){
        long nd2 = in_nodes[indices[offset_edge_start + n]];
        int p_nd2 = partition_map[nd2];
        for(int i=0;i < 4; i++){
          if(i == p_nd2){
            // Denotes node is selected
            ((long *) indptr_map[p_nd1 * 4 + p_nd2])[tid] = 1;
            // Denotes edge is selected
            ((long *)indices_map[p_nd1 * 4 + p_nd2])[offset_edge_start + n] = 1;
          }else{
            // Denotes node is selected
            ((long *)indptr_map[p_nd1 * 4 + p_nd2])[tid] = 0;
            // Denotes edge is selected
            ((long *)indices_map[p_nd1 * 4 + p_nd2])[offset_edge_start + n] = 0;
          }
        }
      }

    //   }
      tid += (blockDim.x * gridDim.x);
    }
}


void Slice::reorder(PartitionedLayer &l){
     // for(int i=0;i < this->num_gpus; i++){
   	 //   l.bipartite[i]->reorder_local(dr, this->num_gpus);
     // }
     // // Handle remote destination nodes
     // for(int to = 0; to < this->num_gpus; to ++){
     //   dr->clear();
     //   dr->order_and_remove_duplicates(l.bipartite[to]->out_nodes_local);
     //   for(int from = 0; from < this->num_gpus; from++){
	   //      if(from == to) continue;
     //     int start = l.bipartite[from]->to_offsets[to];
     //     int end = l.bipartite[from]->to_offsets[to + 1];
     //     l.bipartite[to]->from_ids[from].clear();
     //     vector<long> &t = l.bipartite[to]->from_ids[from];
     //  	 vector<long> &f = l.bipartite[from]->out_nodes_remote;
     //  	 t.insert(t.end(), f.begin() + start, f.begin() + end );
     //  	 dr->replace(t);
     //   }
     // }
     // // Think on paper what I am trying to do here.
     // for(int pull_from = 0;pull_from < this->num_gpus; pull_from++){
     //   dr->clear();
     //   dr->order_and_remove_duplicates(l.bipartite[pull_from]->in_nodes);
     //   for(int pull_to = 0; pull_to < this->num_gpus; pull_to ++ ){
     //     if(pull_from == pull_to)continue;
     //     int start = l.bipartite[pull_to]->pull_from_offsets[pull_from];
     //     int end = l.bipartite[pull_to]->pull_from_offsets[pull_from + 1];
     //     vector<long> &f = l.bipartite[pull_from]->push_to_ids[pull_to];
     //  	 vector<long> &t = l.bipartite[pull_to]->pulled_in_nodes;
     //     assert((end-start) <= t.size());
     //     f.clear();
     //  	 f.insert(f.end(), t.begin() + start, t.begin() + end);
     //     dr->replace(f);
     //   }
     // }
  }

void Slice::slice_layer(thrust::device_vector<long> &layer_nds,
      Block &bs, PartitionedLayer &ps, int partition_id){

    // Stage 1 Edge Partitioning
    ps.resize_index_and_offset_map(layer_nds.size(), bs.indices.size());
    int blocks = (layer_nds.size() - 1)/32 + 1;
    assert(blocks < 620000);
    partition_edges<<<blocks, 32>>>(thrust::raw_pointer_cast(this->workload_map.data()),
        thrust::raw_pointer_cast(layer_nds.data()),
        thrust::raw_pointer_cast(bs.layer_nds.data()),
        thrust::raw_pointer_cast(bs.offsets.data()),
        thrust::raw_pointer_cast(bs.indices.data()),
        (void **)ps.device_offset_map, (void **)ps.device_indices_map, layer_nds.size());

    // Debug Code
    // for(int i = 0; i < 16; i++){
    //    long sum = thrust::reduce(ps.index_offset_map[i].begin(), ps.index_offset_map[i].end(), 0, thrust::plus<long>());
    //    std::cout << i << " " << sum <<"\n";
    // }

    // Stage 2 get sizes of Offsets for all graphs.
    // Inclusive Scan
    for(int i=0; i<16;i ++){
      thrust::inclusive_scan(ps.index_offset_map[i].end(), ps.index_offset_map[i].begin(), ps.index_offset_map[i].end());
      thrust::inclusive_scan(ps.indices_offset_map[i].end(), ps.index_offset_map[i].begin(), ps.index_offset_map[i].end());
    }

    // Stage 3 Populate local and remote edges.

}


void Slice::slice_sample(Sample &s, PartitionedSample &ps){

    for(int i= 1; i< s.num_layers + 1;i++){
	   // std::cout << "Sliceed sample \n";
	    PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
        this->slice_layer(s.block[i-1]->layer_nds, \
          (* s.block[i]), l, layer_id);
	     // this->reorder(l);
    }

    // for(int i=0;i<this->num_gpus;i++){
    //     ps.cache_miss_from[i].clear();
    //     ps.cache_hit_from[i].clear();
    //     ps.cache_miss_to[i].clear();
    //     ps.cache_hit_to[i].clear();
    //     ps.last_layer_nodes[i].clear();
    //       // ps.layers[s.num_layers-1].bipartite[i]->debug();
    //     for(int j = 0; j <ps.layers[s.num_layers- 1].bipartite[i]->in_nodes.size(); j++){
    //       auto nd = ps.layers[s.num_layers-1].bipartite[i]->in_nodes[j];
    //       if (this->storage_map[i][nd] != -1){
    //           ps.cache_hit_from[i].push_back(this->storage_map[i][nd]);
    //           ps.cache_hit_to[i].push_back(j);
    //       }else{
    //         ps.cache_miss_from[i].push_back(nd);
    //         ps.cache_miss_to[i].push_back(j);
    //       }
    //     }
    //     vector<long> &last_layer = ps.layers[0].bipartite[i]->out_nodes_local;
    //     ps.last_layer_nodes[i].insert(ps.last_layer_nodes[i].end(), last_layer.begin(), last_layer.end());
    // }
  }
