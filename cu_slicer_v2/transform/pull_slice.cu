#include "slice.h"
#include <cstring>
#include "../graph/bipartite.h"
#include "../util/cub.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"
#include "../graph/order_book.h"
using namespace cuslicer;

// Elaborate as many cases as possible.
template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void partition_edges_pull(PARTITIONIDX * workload_map, \
      NDTYPE * out_nodes, size_t out_nodes_size, \
    // Sample layer out nodes indexed into the graph
      NDTYPE *in_nodes, size_t in_nodes_size, \
      NDTYPE * indptr, NDTYPE * indices, size_t num_edges, \
      long num_nodes_in_graph, \
        NDTYPE * index_in_nodes_local,
        NDTYPE * index_out_nodes, NDTYPE * index_indptr_local,\
         NDTYPE * index_edge_local,\
      // Partitioned graphs such that indptr_map[dest, src]
       		bool last_layer, OrderBook * book, int NUM_GPUS){
            // Last layer use storage map
    int tileId = blockIdx.x;
    int last_tile = ((out_nodes_size - 1) / TILE_SIZE + 1);
    int work_done = 0;
    while(tileId < last_tile){
        int start = threadIdx.x + (tileId * TILE_SIZE);
        int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),\
            out_nodes_size);
        while(start < end){
            int tid = start;
            auto nd1 = out_nodes[tid];
            auto nbs = indptr[tid+1] - indptr[tid];
            work_done += nbs;
            #ifdef DEBUG
                assert(nd1 < num_nodes_in_graph);
            #endif
            // Sample tid 
            auto p_nd1 = workload_map[tid];
            auto offset_edge_start = indptr[tid];
            index_out_nodes[out_nodes_size * p_nd1 + tid] = 1;
            index_indptr_local[out_nodes_size * p_nd1 + tid] = nbs;

            for(int nb_idx = 0; nb_idx < nbs; nb_idx ++ ){
              auto nd2_idx = indices[offset_edge_start + nb_idx];
        #ifdef DEBUG
            assert(nd2_idx < in_nodes_size);
        #endif
              auto nd2 = in_nodes[nd2_idx];
              auto p_nd2 = workload_map[nd2_idx];

          // In pull optimization always select edge
          ((NDTYPE *)&index_edge_local[p_nd1* num_edges])\
                [offset_edge_start + nb_idx] = 1;
          if(p_nd1 != p_nd2){
                  if(last_layer){
                    if(book->gpuContains(p_nd1, nd2)){
                        index_in_nodes_local[p_nd1 * (in_nodes_size) * NUM_GPUS + nd2_idx] = 1;
                        continue;
                    }
                  }
            auto pull_partition = p_nd2;
            if(p_nd2 > p_nd1)pull_partition = p_nd2 - 1;
            // 1->0,2,3 = 0,1,2 pull_partition ++
            pull_partition ++;
            ((NDTYPE *)&index_in_nodes_local[(p_nd1 * (NUM_GPUS) + pull_partition) * in_nodes_size])[nd2_idx] = 1;
            if(!(nd2_idx < out_nodes_size)){
              // If nd2 idx is less than num out, partition 2 out nodes will
              // mark it anyways
              
                index_in_nodes_local[p_nd2 * (in_nodes_size * NUM_GPUS) + nd2_idx] = 1;
            }
          }else{
            if(!(nd2_idx < out_nodes_size)){
              index_in_nodes_local[p_nd1 * (in_nodes_size * NUM_GPUS) + nd2_idx] = 1;
            }
          }
        }
        start +=  BLOCK_SIZE; 
      }
      tileId += gridDim.x;
    }
    printf("block %d  warp %d threadId %d load %d  \n", blockIdx.x, threadIdx.x /32 , threadIdx.x, work_done);
    
  }

void PullSlicer::resize_bipartite_graphs(PartitionedLayer &ps,\
    int num_in_nodes,\
    int num_out_nodes, int num_edges){
    transform<NDTYPE>::self_inclusive_scan(ps.index_in_nodes);
    transform<NDTYPE>::self_inclusive_scan(ps.index_out_nodes_local);
    transform<NDTYPE>::self_inclusive_scan(ps.index_indptr_local);
    transform<NDTYPE>::self_inclusive_scan(ps.index_edge_local);
    size_t size; size_t offset;
    for(int i=0; i < this->num_gpus; i++){
      BiPartite &bp = *ps.bipartite[i];
      LocalGraphInfo &info = this->host_graph_info[i];

      // Out Nodes Local
      size = ps.index_out_nodes_local[num_out_nodes * (i + 1) - 1];
      offset = 0;
      auto offset_indptr  = 0;
      if( i != 0){
        offset_indptr = ps.index_indptr_local[num_out_nodes * i - 1];
        offset = ps.index_out_nodes_local[num_out_nodes * i - 1];
      }
      size = size - offset;
      bp.out_nodes_local.resize(size);
      bp.num_out_local = size;
      bp.self_ids_offset = size;
      bp.indptr_L.resize(0);
      if(size != 0)bp.indptr_L.resize(size + 1);
      bp.out_degree_local.resize(size);
      info.indptr_L.data = bp.indptr_L.ptr();
      info.indptr_L.offset = offset_indptr;
      info.out_nodes_local.data = bp.out_nodes_local.ptr();
      info.out_nodes_local.offset = offset;
      info.out_degree_local.data = bp.out_degree_local.ptr();
      info.out_degree_local.offset = offset;
      info.num_out_local= size;

      // Local in nodes

      size = ps.index_in_nodes[num_in_nodes * this->num_gpus *  i + num_in_nodes - 1];
      offset = 0;
      if(i != 0){
        offset = ps.index_in_nodes[num_in_nodes * i  * this-> num_gpus - 1];
      }
      size = size - offset;
      bp.in_nodes_local.resize(size + info.num_out_local);
      bp.num_in_nodes_local = size + info.num_out_local  ;
      info.in_nodes_local.data = bp.in_nodes_local.ptr();
      info.in_nodes_local.offset = offset;

      // Local pull nodes
      // Add pull offsets. 
      //  Review not clean 
       bp.pull_from_offsets[0] = 0;
       bp.to_offsets[0] = 0;
       auto in_not_pulled = ps.index_in_nodes[(num_in_nodes * this->num_gpus * i) + num_in_nodes - 1];
       for(int from = 0; from < this->num_gpus; from ++){
        bp.to_offsets[from + 1] = 0;
        if(i == from ){bp.pull_from_offsets[from + 1] = bp.pull_from_offsets[i]; continue;}
        auto l_from = from;
        if (from > i) l_from = from - 1;
        bp.pull_from_offsets[from + 1 ] \
        = ps.index_in_nodes[(num_in_nodes  * this-> num_gpus ) * i + num_in_nodes + num_in_nodes * (l_from + 1) - 1]\
                - in_not_pulled ;
        ps.bipartite[from]->pull_to_ids[i].resize(bp.pull_from_offsets[from+1] - bp.pull_from_offsets[from]);
        // bp.pull_from_offsets[from + 1] +=bp.pull_from_offsets[from];
        this->host_graph_info[from].pull_to_ids[i].data = ps.bipartite[from]->pull_to_ids[i].ptr();
        this->host_graph_info[from].pull_to_ids[i].offset = \
                ps.index_in_nodes[(num_in_nodes  * this-> num_gpus * i ) + num_in_nodes +  (num_in_nodes * l_from) - 1];
                std::cout << "resize offset " << this->host_graph_info[from].pull_to_ids[i].offset <<"\n";  
       }  
       bp.num_in_nodes_pulled  = bp.pull_from_offsets[this-> num_gpus] ;
       std::cout << "Pulled node !!!!!!" ;
       for(int ii = 0; ii < this->num_gpus + 1; ii++){
          std::cout << bp.pull_from_offsets[ii] << " ";
       }
       std::cout <<"\n";
      // Edges Local 
      std::cout << "edges local\n";
      offset = 0;
      if(i !=0 ){
        offset = ps.index_edge_local[num_edges * i - 1];
      }
      auto end = ps.index_edge_local[num_edges * (i + 1) - 1];
      bp.indices_L.resize(end - offset);
      info.indices_L.data = bp.indices_L.ptr();
      info.indices_L.offset = offset;
    }
     copy_graph_info();
}

template<int BLOCKSIZE, int TILESIZE>
__global__
void fill_out_nodes_pull(NDTYPE * index_out_nodes_local,\
        size_t index_out_nodes_local_size,\
       // mask is set to where to write
        NDTYPE * index_indptr_local, \
        LocalGraphInfo *info, int num_gpus,\
        // Meta data
        NDTYPE *out_nodes, NDTYPE * out_node_degree,\
        long num_out_nodes){
        int tileId = blockIdx.x;
        int last_tile = (( index_out_nodes_local_size - 1) / TILE_SIZE + 1);
        while(tileId < last_tile){
        int start = threadIdx.x + (tileId * TILE_SIZE);
        int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), index_out_nodes_local_size);
        while(start < end){
            int tid = start;

            int gpu_id = tid/num_out_nodes;
            long out_node_idx = tid % num_out_nodes;
            
            if(is_selected(index_out_nodes_local, tid)){
               int write_index =  index_out_nodes_local[tid];
               
               assert(write_index > 0);
               // Fill indptr
               auto remote_pos = write_index - info[gpu_id].out_nodes_local.offset;
                if(remote_pos == 1){
                  info[gpu_id].indptr_L.data[0] = 0;
                }
	             info[gpu_id].indptr_L.add_value_offset(index_indptr_local[tid],remote_pos);
               info[gpu_id].out_nodes_local.add_position_offset( out_nodes[out_node_idx],write_index);
               info[gpu_id].out_degree_local.add_position_offset(out_node_degree[out_node_idx],write_index);
             }
            start += BLOCK_SIZE;
      }
      tileId += gridDim.x;
    }
}

template<int BLOCKSIZE, int TILESIZE>
__global__ void fill_in_nodes_pull(NDTYPE * index_in_nodes, \
    NDTYPE * index_out_nodes_local, \
    LocalGraphInfo *info, int num_gpus,\
      NDTYPE * in_nodes, size_t num_in_nodes,
      size_t num_out_nodes){
        int tileId = blockIdx.x;
        int lastTile = ( num_in_nodes * num_gpus * num_gpus- 1)/TILE_SIZE + 1;
        while(tileId < lastTile){
        
        int start = threadIdx.x + (tileId * TILE_SIZE);
        int end = min(static_cast<int64_t>(threadIdx.x +\
         (tileId + 1) * TILE_SIZE), num_in_nodes * num_gpus * num_gpus);
        while(start < end){
          int tid = start;
          int gpu_id = tid / (num_in_nodes * num_gpus);
          int pull_from_gpu = (tid / num_in_nodes ) % num_gpus;
          auto in_node_idx = tid % num_in_nodes;
              // is a self node
            if(in_node_idx < num_out_nodes){
              if(is_selected(index_out_nodes_local,\
               (num_out_nodes * gpu_id + in_node_idx))){
                  long in_node = in_nodes[tid % num_in_nodes];
                  auto d = info[gpu_id].out_nodes_local;
                  auto write_index = index_out_nodes_local[num_out_nodes * gpu_id + in_node_idx] - d.offset - 1;
                  info[gpu_id].in_nodes_local.data[write_index] = in_node;
                  start += BLOCK_SIZE;
                  continue;
              }
            }
            if(is_selected(index_in_nodes, tid)  && (pull_from_gpu == 0)){
              long in_node = in_nodes[tid % num_in_nodes];
              // num out local nodes are not marked in in nodes
              auto write_index = index_in_nodes[tid] \
              + info[gpu_id].num_out_local;
              info[gpu_id].in_nodes_local.\
              add_position_offset(in_node, write_index);
            }
            if(is_selected(index_in_nodes, tid) && (pull_from_gpu != 0)){
              long in_node = in_nodes[tid % num_in_nodes];
              auto write_index = index_in_nodes[tid] ;
              //1,2,3 = 0,1,2 for for all gpu. 
              // because pull from gpu is always > 1 
              if(pull_from_gpu <= gpu_id)pull_from_gpu --;
              info[pull_from_gpu].pull_to_ids[gpu_id]\
                .add_position_offset(in_node,write_index);
          }
        start += BLOCK_SIZE;
      }
    tileId += gridDim.x;
  }
}

template<int BLOCKSIZE, int TILESIZE>
__global__ void fill_indices_local_pull(NDTYPE *sample_indices,
      NDTYPE *index_edges, size_t num_edges,
      NDTYPE * index_in_nodes, size_t num_in_nodes,
      NDTYPE * index_out_nodes_local, size_t num_out_nodes,
      LocalGraphInfo *info, int num_gpus){
  int tileId = blockIdx.x;
  int last_tile = (((num_edges * num_gpus) - 1) / TILE_SIZE + 1);
  while(tileId < last_tile){
  int start = threadIdx.x + (tileId * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), num_edges * num_gpus);
  while(start < end){
    int tid = start;
    int gpu_id = tid / (num_edges);
    if(is_selected(index_edges,tid)){
       auto edge_idx = tid % num_edges;
       long indice;
       if((sample_indices[edge_idx] < num_out_nodes) && \
        is_selected(index_out_nodes_local,\
           gpu_id * num_out_nodes + sample_indices[edge_idx])){
         indice = index_out_nodes_local\
         [num_out_nodes * gpu_id + sample_indices[edge_idx]] \
              - info[gpu_id].out_nodes_local.offset - 1;
        assert(indice >= 0);
       }else{
          auto in_node_idx = sample_indices[edge_idx];
          // either a local node or a pulled node.
          indice = -10;
          for(int j = 0; j < num_gpus; j++){
            if(is_selected(index_in_nodes, (num_gpus * gpu_id + j) * num_in_nodes + in_node_idx)){
              indice = info[gpu_id].num_out_local +\
          index_in_nodes\
          [(num_gpus * gpu_id + j) * num_in_nodes + in_node_idx]\
           - info[gpu_id].in_nodes_local.offset - 1;
            assert(indice>= 0);
            break;
            }
          }
          
          assert(indice >= 0);
          // What does this block do ?
         
        }
        assert(indice != -1);
       info[gpu_id].indices_L.add_position_offset(indice, index_edges[tid]);
      }
    start += BLOCK_SIZE;
  }
    tileId += gridDim.x;
  }
}

void PullSlicer::slice_layer(device_vector<NDTYPE> &layer_nds,
      Block &bs, PartitionedLayer &ps, bool last_layer){
    
    // Stage 1 Edge Partitioning
    ps.resize_selected_pull\
        (layer_nds.size(), bs.indices.size(),\
         bs.layer_nds.size());

    auto num_out_nodes = layer_nds.size();
    auto num_edges = bs.indices.size();
    auto num_in_nodes = bs.layer_nds.size();
    std::cout << "Launch configuration " << GRID_SIZE(layer_nds.size()) << " " << TILE_SIZE <<"\n";
    partition_edges_pull<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(layer_nds.size()),BLOCK_SIZE>>>\
        ( this->sample_workload_map.ptr(),\
          layer_nds.ptr(), layer_nds.size(),\
          bs.layer_nds.ptr(), bs.layer_nds.size(),\
          bs.offsets.ptr(),bs.indices.ptr(), bs.indices.size(),\
          this->num_nodes,\
          ps.index_in_nodes.ptr(),\
          ps.index_out_nodes_local.ptr(), ps.index_indptr_local.ptr(),
          ps.index_edge_local.ptr(),\
          last_layer, this->orderbook->getDevicePtr(),this->num_gpus);
    gpuErrchk(cudaDeviceSynchronize());
    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif
    // ps.index_in_nodes.debug("Index in nodes local !!!!!!!!");
    // ps.index_out_nodes_local.debug("Out nodes local check");
    // Stage 2 get sizes of Offsets for all graphs
    // Inclusive Scan
    this->resize_bipartite_graphs(ps,\
       num_in_nodes, num_out_nodes, num_edges);
    // Stage 3 Populate local and remote edges.
    // Populate graph local in nodes
     fill_in_nodes_pull<BLOCK_SIZE, TILE_SIZE>\      
        <<<GRID_SIZE( ps.index_in_nodes.size()),BLOCK_SIZE>>>(ps.index_in_nodes.ptr(), \
        ps.index_out_nodes_local.ptr(), \
       this->device_graph_info, num_gpus, bs.layer_nds.ptr(), num_in_nodes, num_out_nodes);

    // Populate pulled in nodes 
    // std::cout << "num edges" << num_edges <<":"<< ps.index_edge_local.size() << "\n";
    // ps.index_edge_local.debug("edge local check ");
    // ps.index_edge_local.debug("Edge local");
    // ps.index_in_nodes.debug("In node local");
    // ps.index_out_nodes_local.debug("Out node local");
    
    // populate edges
    // gpuErrchk(cudaDeviceSynchronize());
    fill_indices_local_pull<BLOCK_SIZE, TILE_SIZE>\
    <<<GRID_SIZE( ps.index_edge_local.size()),BLOCK_SIZE>>>(bs.indices.ptr(),
         ps.index_edge_local.ptr(), num_edges, \
         ps.index_in_nodes.ptr() , num_in_nodes, \
         ps.index_out_nodes_local.ptr(), num_out_nodes, \
          this->device_graph_info, num_gpus);
         #ifdef DEBUG
           gpuErrchk(cudaDeviceSynchronize());
         #endif
    // Replace pull indices correctly
    // 3 functions do everything.   
    // Out Nodes
    // std::cout << "Check till here\n";
    // ps.index_out_nodes_local.debug("out nodes local");
    // ps.index_indptr_local.debug("indptr loca");
    fill_out_nodes_pull<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(num_out_nodes), TILE_SIZE>>>(\
        ps.index_out_nodes_local.ptr(),\   
        ps.index_out_nodes_local.size(),\
        // masks set to where to write
        ps.index_indptr_local.ptr(), \
        this->device_graph_info,  num_gpus, 
        layer_nds.ptr(), bs.in_degree.ptr(),num_out_nodes);
    
    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif    

    ps.resize_selected_pull\
        (0,0,0);

    std::cout << "slice successful !!\n";
}
