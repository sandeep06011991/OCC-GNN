#include "transform/slice.h"
#include <cstring>
#include "graph/bipartite.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

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

          // In pull optimization always select edge
         ((long *)indices_map[p_nd1 * NUM_GPUS + p_nd1])[offset_edge_start + nb_idx] = 1;
         if(p_nd1 != p_nd2){
           if(last_layer){
         		if(((int *)storage_map[p_nd1])[nd2]!= -1){
               // Non partitioned node is present locally.
         			 continue;
         		}
         	}
          ((long *)indices_map[p_nd1 * NUM_GPUS + p_nd2])[offset_edge_start + nb_idx] = 1;
         }

       }
      ((long *) indptr_map[p_nd1 * NUM_GPUS + p_nd1])[tid] = 1;
       tid += (blockDim.x * gridDim.x);
    }
}

__global__ void populate_local_graphs_pull(int*  partition_map, long * out_nodes,
      long *in_nodes, long * indptr, long *indices,\
        void ** indptr_index_map, void ** indices_index_map,
         void ** indptr_map, void ** indices_map,
            void ** from_nds_map, void ** out_degree_map,  int size,\
	    	bool last_layer, void ** storage_map, int NUM_GPUS){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while(tid < size){
      long nd1 = out_nodes[tid];
      long nbs = indptr[tid+1] - indptr[tid];
      int p_nd1 = partition_map[nd1];
      long offset_edge_start = indptr[tid];

      for(int n = 0; n<nbs; n ++ ){
          long nd2 = in_nodes[indices[offset_edge_start + n]];
          int p_nd2 = partition_map[nd2];
          ((long *)indices_map[p_nd1 * NUM_GPUS + p_nd1])[offset_edge_start + n] = nd2;
           if(p_nd1 != p_nd2){
           if(last_layer){
                  if(((int *)storage_map[p_nd1])[nd2]!= -1){
                        continue;
                  }
          }
          ((long *)from_nds_map[p_nd1 * NUM_GPUS + p_nd2])\
                  [((long *)indices_index_map[p_nd1 * NUM_GPUS + p_nd2])[offset_edge_start + n]] \
                  = nd2;
          }
        }
     ((long *)indptr_map[p_nd1 * NUM_GPUS + p_nd1])\
        [((long *)indptr_index_map[p_nd1 * NUM_GPUS + p_nd1])[tid]] = nbs;

        if(nbs == 0) nbs = 1;
        ((long *)out_degree_map[p_nd1])\
            [((long *)indptr_index_map[p_nd1 * NUM_GPUS + p_nd1])[tid] - 1] = nbs;
      tid += (blockDim.x * gridDim.x);
    }
}



void PullSlicer::slice_layer(thrust::device_vector<long> &layer_nds,
      Block &bs, PartitionedLayer &ps, bool last_layer){

    // Stage 1 Edge Partitioning
    ps.resize_index_and_offset_map(layer_nds.size(), bs.indices.size());
      partition_edges_pull<<<BLOCK_SIZE(layer_nds.size()),THREAD_SIZE>>>\
        (thrust::raw_pointer_cast(this->workload_map.data()),\
          thrust::raw_pointer_cast(layer_nds.data()), layer_nds.size(),\
          thrust::raw_pointer_cast(bs.layer_nds.data()), bs.layer_nds.size(),\
          thrust::raw_pointer_cast(bs.offsets.data()),\
          thrust::raw_pointer_cast(bs.indices.data()),\
          this->num_nodes,\
            (void **)ps.device_offset_map, (void **)ps.device_indices_map,\
          last_layer, this->storage_map_flattened,this->num_gpus);

    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif
    // Stage 2 get sizes of Offsets for all graphs
    // Inclusive Scan
    int N = this->num_gpus * this->num_gpus;
    long local_graph_nodes[N];
    long local_graph_edges[N];
    for(int i=0; i< N; i++){
      long last_indices = ps.index_indices_map[i][ps.index_indices_map[i].size() - 1];
      thrust::inclusive_scan(ps.index_offset_map[i].begin(), ps.index_offset_map[i].end(), ps.index_offset_map[i].begin());
      thrust::exclusive_scan(ps.index_indices_map[i].begin(), ps.index_indices_map[i].end(), ps.index_indices_map[i].begin());
      local_graph_nodes[i] = ps.index_offset_map[i][ps.index_offset_map[i].size()-1];
      local_graph_edges[i] = ps.index_indices_map[i][ps.index_indices_map[i].size() - 1] + last_indices;
      std::cout << "G" <<  local_graph_nodes[i] <<":" <<local_graph_edges[i] <<"\n";
      }
    ps.resize_local_graphs(local_graph_nodes, local_graph_edges);
    // Stage 3 Populate local and remote edges.

    populate_local_graphs_pull<<<BLOCK_SIZE(layer_nds.size()), THREAD_SIZE>>>\
      (thrust::raw_pointer_cast(this->workload_map.data()),\
        thrust::raw_pointer_cast(layer_nds.data()),
        thrust::raw_pointer_cast(bs.layer_nds.data()),
        thrust::raw_pointer_cast(bs.offsets.data()),
        thrust::raw_pointer_cast(bs.indices.data()),
        (void **)ps.device_offset_map,
        (void **)ps.device_indices_map,
        (void **)ps.device_local_indptr_map,
        (void **)ps.device_local_indices_map,
        (void **)ps.device_local_pull_from_nds_map,
        (void **)ps.device_out_nodes_degree_map,
        layer_nds.size(),\
        last_layer, this->storage_map_flattened, this->num_gpus);
    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif
}
