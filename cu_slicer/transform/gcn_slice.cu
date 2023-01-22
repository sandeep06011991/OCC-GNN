#include "transform/slice.h"
#include <cstring>
#include "graph/bipartite.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// void debug_vector(thrust::device_vector<long>& ls){
//   std::cout << "DEBUG:";
//   for(int i = 0;i<ls.size();i++){
//       std::cout << ls[i] << " ";
//   }
//   std::cout <<"\n";
// }
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
      int p_nbs [4];
      for(int n=0; n<4; n++){
        p_nbs[n] = 0;
      }
      p_nbs[nd1] = 1;
      // ((long *) indptr_map[p_nd1 * 4 + p_nd1])[tid] = 1;

      for(int n = 0; n<nbs; n ++ ){
        long nd2 = in_nodes[indices[offset_edge_start + n]];

        int p_nd2 = partition_map[nd2];
        // printf("%ld %ld %d %d %d\n", nd1, nd2, tid, p_nd1, p_nd2);
        // Always select out node for its own partition even if it has no outgoing edge
        p_nbs[p_nd2] = 1;
        for(int i=0;i < 4; i++){
          if(i == p_nd2){
            // Denotes node is selected
            // Denotes edge is selected
            // ((long *)indices_map[p_nd1 * 4 + p_nd2])[offset_edge_start + n] = 1;
          }else{
            // Denotes node is selected
            // ((long *)indptr_map[p_nd1 * 4 + p_nd2])[tid] = 0;
            // Denotes edge is selected
            // ((long *)indices_map[p_nd1 * 4 + p_nd2])[offset_edge_start + n] = 0;
          }
        }
      }
      for(int p_nd2 = 0; p_nd2<4 ;p_nd2++){
        if(p_nbs[p_nd2]){
          assert(tid < size);
          assert(p_nd1 * 4 + p_nd2 < 16);
          ((long *) indptr_map[p_nd1 * 4 + p_nd2])[tid] = 1;
        }
      }
      tid += (blockDim.x * gridDim.x);
    }
}

__global__ void populate_local_graphs(int*  partition_map, long * out_nodes,
      long *in_nodes, long * indptr, long *indices,\
        void ** indptr_index_map, void ** indices_index_map,
         void ** indptr_map, void ** indices_map,
            void ** to_nds_map, void ** out_degree_map,  int size){

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

void Slice::reorder(PartitionedLayer &l){
    for(int i=0;i < this->num_gpus; i++){
      std::cout << "local reordering " << i <<"\n";
       l.bipartite[i]->reorder_local(dr);
     }
     // Handle remote destination nodes
     for(int to = 0; to < this->num_gpus; to ++){
       std::cout << "Handle remote nodes \n";
       dr->clear();
       dr->order_and_remove_duplicates(l.bipartite[to]->out_nodes_local);
       for(int from = 0; from < this->num_gpus; from++){
	        if(from == to) continue;
         int start = l.bipartite[from]->to_offsets[to];
         int end = l.bipartite[from]->to_offsets[to + 1];
         l.bipartite[to]->from_ids[from].clear();
         thrust::device_vector<long> &t = l.bipartite[to]->from_ids[from];
      	 thrust::device_vector<long> &f = l.bipartite[from]->out_nodes_remote;
      	 t.insert(t.end(), f.begin() + start, f.begin() + end );
      	 dr->replace(t);
       }
     }

  }

void Slice::slice_layer(thrust::device_vector<long> &layer_nds,
      Block &bs, PartitionedLayer &ps, int partition_id){

    // Stage 1 Edge Partitioning
    std::cout << "edge partitioning " << layer_nds.size() << " "<< bs.offsets.size() <<" "<< bs.indices.size() << " "<<bs.layer_nds.size() <<  "\n";
    ps.resize_index_and_offset_map(layer_nds.size(), bs.indices.size());
    int blocks = (layer_nds.size() - 1)/32 + 1;
    assert(blocks < 620000);
    partition_edges<<<blocks, 32>>>(thrust::raw_pointer_cast(this->workload_map.data()),
        thrust::raw_pointer_cast(layer_nds.data()),
        thrust::raw_pointer_cast(bs.layer_nds.data()),
        thrust::raw_pointer_cast(bs.offsets.data()),
        thrust::raw_pointer_cast(bs.indices.data()),
        (void **)ps.device_offset_map, (void **)ps.device_indices_map, layer_nds.size());
    gpuErrchk(cudaDeviceSynchronize());
    // Debug Code
    // for(int i = 0; i < 16; i++){
    //    long sum = thrust::reduce(ps.index_offset_map[i].begin(), ps.index_offset_map[i].end(), 0, thrust::plus<long>());
    //    std::cout << i << " " << sum <<"\n";
    // }

    // Stage 2 get sizes of Offsets for all graphs
    // Inclusive Scan
    std::cout << "local graph sizeing \n";
    long local_graph_nodes[16];
    long local_graph_edges[16];
    // for(int i=0; i<16;i ++){
    //   for(int j = 0; j<ps.index_offset_map[i].size(); j++){
    //     std::cout << ps.index_offset_map[i][j] <<" ";
    //   };
    //   std::cout << "Running " <<ps.index_offset_map[i].size() <<":" <<  ps.index_indices_map[i].size() << "\n";
    //   thrust::inclusive_scan(ps.index_offset_map[i].begin(), ps.index_offset_map[i].end(), ps.index_offset_map[i].begin());
    //   std::cout << "DOne " <<ps.index_offset_map[i].size() <<":" <<  ps.index_indices_map[i].size() << "\n";
    //   thrust::inclusive_scan(ps.index_indices_map[i].begin(), ps.index_indices_map[i].end(), ps.index_indices_map[i].begin());
    //   std::cout << "DOne " <<ps.index_offset_map[i].size() <<":" <<  ps.index_indices_map[i].size() << "\n";
    //   local_graph_nodes[i] = ps.index_offset_map[i][ps.index_offset_map[i].size()-1];
    //
    //   local_graph_edges[i] = ps.index_indices_map[i][ps.index_indices_map[i].size() - 1];
    // }
    // ps.resize_local_graphs(local_graph_nodes, local_graph_edges);
    // Stage 3 Populate local and remote edges.
    std::cout << "local graph partitioning \n";
    // populate_local_graphs<<<blocks, 32>>>(thrust::raw_pointer_cast(this->workload_map.data()),
    //     thrust::raw_pointer_cast(layer_nds.data()),
    //     thrust::raw_pointer_cast(bs.layer_nds.data()),
    //     thrust::raw_pointer_cast(bs.offsets.data()),
    //     thrust::raw_pointer_cast(bs.indices.data()),
    //     (void **)ps.device_offset_map,
    //     (void **)ps.device_indices_map,
    //     (void **)ps.device_local_indptr_map,
    //     (void **)ps.device_local_indices_map,
    //     (void **)ps.device_local_to_nds_map,
    //     (void **)ps.device_out_nodes_degree_map,
    //     layer_nds.size());
    cudaDeviceSynchronize();
}


void Slice::slice_sample(Sample &s, PartitionedSample &ps){

    // for(int i= 1; i< s.num_layers + 1;i++){
    for(int i=2;i<3;i++){
	   // std::cout << "Sliceed sample \n";
	    PartitionedLayer& l = ps.layers[i-1];
        int layer_id = i-1;
        std::cout << "Slicing starts" <<"\n";
        this->slice_layer(s.block[i-1]->layer_nds, \
          (* s.block[i]), l, layer_id);
       //  std::cout << "Slicing ends" <<"\n";
	     // this->reorder(l);
       // std::cout << "Reordering starts \n";

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
