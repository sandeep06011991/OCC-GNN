
#include "slice.h"
#include <cstring>
#include "../graph/bipartite.h"
#include "../util/cub.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"
#include "../graph/order_book.h"
using namespace cuslicer;


/*
* workload_map is the size of all the nodes in the graph 
*/
// template<int BLOCK_SIZE, int TILE_SIZE, int SUBWARPPERNODE>
// __global__
// void partition_edges_redundant(PARTITIONIDX * workload_map, size_t, workmap, device_Block block, \
//   Dataset graph,  LocalGraphInfo info,  int NUM_GPUS){
//     int tileId = blockIdx.x;
//     int last_tile = ((out_nodes_size - 1) / TILE_SIZE + 1);
//     while(tileId < last_tile){
//         int start = threadIdx.x + (tileId * TILE_SIZE);
//         int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),\
//             out_nodes_size);
//         while(start < end){
//             int tid = start;
//             auto nd1 = out_nodes[tid];
//             auto nbs = indptr[tid+1] - indptr[tid];
// //             auto p_nd1 = workload_map[tid];
// //             auto offset_edge_start = indptr[tid];
//             index_out_nodes[out_nodes_size * p_nd1 + tid] = 1;
//             index_indptr_local[out_nodes_size * p_nd1 + tid] = nbs;

//             for(int nb_idx = warp_id; nb_idx < nbs; nb_idx + SUBWARPPERNODE ){
//               auto nd2_idx = indices[offset_edge_start + nb_idx];
// //         #ifdef DEBUG
// //             assert(nd2_idx < in_nodes_size);
// //         #endif
// //               auto nd2 = in_nodes[nd2_idx];
// //               auto p_nd2 = workload_map[nd2_idx];

// //           // In pull optimization always select edge
//           ((NDTYPE *)&index_edge_local[p_nd1* num_edges])\
//                 [offset_edge_start + nb_idx] = 1;
//            ((NDTYPE *)&index_in_nodes_local[(p_nd1 * (NUM_GPUS) + pull_partition) * in_nodes_size])[nd2_idx] = 1;
//         }
//         start +=  BLOCK_SIZE; 
//       }
//       tileId += gridDim.x;
//     }
    
//   }

  // Get all offsets 

  // Resize the whole graph 


// void FusedSlicer::slice_layer(device_vector<NDTYPE> &layer_nds,
//       Block &bs, PartitionedLayer &ps, bool last_layer){
        
//       if(!last_layer){
//         // partition_edges_redundant 

//         // resize all bipartite graphs 

//         // Fill all nodes
//         for(int i=0; i < this->num_gpus; i++){
//             BiPartite &bp = *ps.bipartite[i];
//             bp.gpu_id = i;
//             bp.num_gpus = this->num_gpus;
//             bp.in_nodes_local = bs.layer_nds;
//             bp.out_nodes_local = layer_nds;
//             bp.indptr_L = bs.offsets;
//             bp.indices_L = bs.indices;
//             bp.out_degree_local = bs.in_degree;
//             bp.indptr_R.clear();
//             bp.indices_R.clear();
//             bp.pulled_in_nodes.clear();
//             bp.pull_from_offsets[0] = 0;
//             bp.to_offsets[0] = 0;
//             for(int j = 0; j < this->num_gpus; j ++ ){
//                 bp.pull_from_offsets[j+1] = 0;
//                 bp.to_offsets[j+1] = 0;
//                 bp.pull_from_ids_[j].clear();
//                 bp.push_to_ids_[j].clear();
//             }
//          }
//       }else{
//         PullSlicer::slice_layer(layer_nds, bs, ps, last_layer);
//         // call pull slicer code 
//       }
//     }