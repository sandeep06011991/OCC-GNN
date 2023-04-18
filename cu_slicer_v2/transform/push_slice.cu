#include "slice.h"
#include <cstring>
#include "../graph/bipartite.h"
#include "nvtx3/nvToolsExt.h"
#include "../util/cub.h"

using namespace cuslicer;


template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void partition_edges_push(int*  partition_map,\ // partition_map assigning each vertex ID to one GPU
    long * out_nodes, size_t out_nodes_size,\ // Sample layer out nodes indexed into the graph
      long *in_nodes, size_t in_nodes_size, \ // Sample Layer In Nodes Indexed into the graph
        long * indptr, long *indices, size_t num_edges, \// Sampled graph representation
          long num_nodes_in_graph, \
            long * index_in_nodes, long * index_out_nodes_local,\
             long * index_out_nodes_remote,\
              long * index_indptr_local, long * index_indptr_remote, \
                long * index_edge_local, long * index_edge_remote,\
      // Partitioned graphs such that indptr_map[dest, src]
       	  	bool last_layer, void ** storage_map, int NUM_GPUS){
            // Last layer use storage map
    int tileId = blockIdx.x;
    int last_tile = ((out_nodes_size - 1) / TILE_SIZE + 1);
    while(tileId < last_tile){
    int start = threadIdx.x + (tileId * TILE_SIZE);
    int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), out_nodes_size);
    while(start < end){
      int tid = start;
      long nd1 = out_nodes[tid];
      long nbs = indptr[tid+1] - indptr[tid];
      #ifdef DEBUG
          assert(nd1 < num_nodes_in_graph);
      #endif
	//int p_nd1 = 0;
      int p_nd1 = partition_map[nd1];
      long offset_edge_start = indptr[tid];
      int p_nbs[MAX_DEVICES];
      for(int n=0; n<NUM_GPUS; n++){
        p_nbs[n] = 0;
      }
      p_nbs[p_nd1] = 0;
      for(int nb_idx = 0; nb_idx < nbs; nb_idx ++ ){
        long nd2_idx = indices[offset_edge_start + nb_idx];
        #ifdef DEBUG
            assert(nd2_idx < in_nodes_size);
        #endif
        long nd2 = in_nodes[nd2_idx];
	//int p_nd2 = 0;
	int p_nd2 = partition_map[nd2];
        if(p_nd1 == p_nd2){
          // Same partition add local edge
          ((long *)&index_edge_local[num_edges * p_nd1])[offset_edge_start + nb_idx] = 1;
          // Mark only if not in the outnode list
          // Makes setting self nodes easier
          if(!(nd2_idx < out_nodes_size)){
            ((long *)&index_in_nodes[in_nodes_size * p_nd2])[nd2_idx] = 1;
            }
           p_nbs[p_nd1] ++;
           continue;
        }
      	if(last_layer){
          // Not the same partition but part of our redundant store.
      		if(((int *)storage_map[p_nd1])[nd2]!= -1){
                  // Present here mark it local.
      			      ((long *)&index_edge_local[num_edges * p_nd1])[offset_edge_start + nb_idx] = 1;
                  ((long *)&index_in_nodes[in_nodes_size * p_nd1])[nd2_idx] = 1;
                  p_nbs[p_nd1] ++;
             continue;
    		   }
      	}
        // ToDO keep note of when to mark and when to not mark
        // local nodes
        //  remote edge
        // remote partitions are stored continuosly
        // For ex. remote edge for gpu 2 =[0, 1, 3].
        // Adjusting that third partition is at the second index.
        int remote_offset= p_nd1;
        if(p_nd1 > p_nd2) remote_offset = remote_offset-1;

        ((long *)&index_edge_remote[num_edges * (NUM_GPUS - 1) * p_nd2  +  remote_offset  * num_edges])[offset_edge_start + nb_idx] = 1;
        // change back value This is incorrect
        if(!(nd2_idx < out_nodes_size)){
          ((long *)&index_in_nodes[in_nodes_size * p_nd2])[nd2_idx] = 1;
        }
        p_nbs[p_nd2] ++;
      }

      for(int p_nd = 0; p_nd< NUM_GPUS ;p_nd++){
        // p_nd denotes partition with atleast one remote outgoing edge
          if(p_nd == p_nd1){
            ((long *)&index_out_nodes_local[out_nodes_size * p_nd])[tid] = 1;
            ((long *)&index_indptr_local[out_nodes_size * p_nd])[tid] = p_nbs[p_nd];
            // Dont mark out nodes in innodes
            // ((long *)&index_in_nodes[in_nodes_size * p_nd])[tid] = 1;
          }else{
          if(p_nbs[p_nd] > 0){
            int remote_offset= p_nd1;
            if(p_nd1 > p_nd) remote_offset = remote_offset-1;
            ((long *)&index_out_nodes_remote[out_nodes_size * (NUM_GPUS - 1) * p_nd  + remote_offset * out_nodes_size])[tid] = 1;
            ((long *)&index_indptr_remote[out_nodes_size * (NUM_GPUS - 1) * p_nd  + remote_offset *out_nodes_size])[tid] = p_nbs[p_nd];
          }
      }
    }
      start +=  BLOCK_SIZE;
  }
      tileId += gridDim.x;
  }
}

__device__
bool is_selected(long *id, size_t sz){
   if(sz == 0)return id[0] != 0;
   return id[sz] != id[sz-1];
}

template<int BLOCKSIZE, int TILESIZE>
__global__ void fill_indices_local(long *sample_indices,
      long *index_edges, long num_edges,
      long * index_in_nodes, size_t num_in_nodes,
      long * index_out_nodes_local, size_t num_out_nodes,
      PushSlicer::LocalGraphInfo *info, int num_gpus){
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
        is_selected(index_out_nodes_local, gpu_id * num_out_nodes + sample_indices[edge_idx])){
         indice = index_out_nodes_local[num_out_nodes * gpu_id + sample_indices[edge_idx]] \
              - info[gpu_id].out_nodes_local.offset - 1;

       }else{
         indice = info[gpu_id].num_out_local +\
          index_in_nodes[num_in_nodes * gpu_id + sample_indices[tid % num_edges]] - info[gpu_id].in_nodes.offset - 1;
        }
       info[gpu_id].indices_L.add_position_offset(indice, index_edges[tid]);
      }
    start += BLOCK_SIZE;
  }
    tileId += gridDim.x;
  }
}

template<int BLOCKSIZE, int TILESIZE>
__global__ void fill_indices_remote(long * index_edge_remote, size_t num_edges, \
       long * index_in_nodes, size_t num_in_nodes,
           long * index_out_nodes_local, size_t num_out_nodes,
              PushSlicer::LocalGraphInfo * info, int num_gpus, long *sample_indices){
    int tileId = blockIdx.x;
    int last_tile = (((num_edges * (num_gpus - 1) * (num_gpus)) - 1) / TILE_SIZE + 1);
    while(tileId < last_tile){
    int start = threadIdx.x + (tileId * TILE_SIZE);
    int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), num_edges * (num_gpus - 1) * (num_gpus));
    while(start < end){
      int tid = start;
      int gpu_id = tid / (num_edges * (num_gpus - 1));
      if(is_selected(index_edge_remote,tid) ){
        auto edge_idx = tid % num_edges;
        long indice;
        if((sample_indices[edge_idx] < num_out_nodes) && \
          is_selected(index_out_nodes_local, gpu_id * num_out_nodes + sample_indices[edge_idx])){
          indice = index_out_nodes_local[num_out_nodes * gpu_id + sample_indices[edge_idx]] \
               -info[gpu_id].out_nodes_local.offset - 1;

        }else{
          indice = info[gpu_id].num_out_local +\
            index_in_nodes[num_in_nodes * gpu_id +sample_indices[tid % num_edges]] - info[gpu_id].in_nodes.offset - 1;
        }
        info[gpu_id].indices_R.add_position_offset(indice, index_edge_remote[tid]);
      }
      start += BLOCK_SIZE;
    }
      tileId += gridDim.x;
    }
}

template<int BLOCKSIZE, int TILESIZE>
__global__
void fill_out_nodes_local(long * index_out_nodes_local,\
      size_t index_out_nodes_local_size,\  // mask is set to where to write
        long * index_indptr_local, \
        PushSlicer::LocalGraphInfo *info, int num_gpus,\ // Meta data
        long *out_nodes, long * out_node_degree,\
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
            if(out_node_idx == 0){
                // info[gpu_id].indptr_L.data[0] = 0;
            }
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
__global__ void fill_out_nodes_remote(long * index_out_nodes_remote, \
        size_t index_out_nodes_remote_sizes, \
        long * index_indptr_remote, \
        PushSlicer::LocalGraphInfo *info, int num_gpus,\ // Meta data
        long *out_nodes, long num_out_nodes ){
          int tileId = blockIdx.x;
          int last_tile = (( index_out_nodes_remote_sizes - 1) / TILE_SIZE + 1);
          while(tileId < last_tile){
          int start = threadIdx.x + (tileId * TILE_SIZE);
          int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE),  index_out_nodes_remote_sizes);
          while(start < end){
            int tid = start;
            int gpu_idx = tid / num_out_nodes;
            int from_gpu_id = gpu_idx / (num_gpus - 1);
            int to_gpu_idx =  gpu_idx % (num_gpus - 1);
            int to_gpu_id = to_gpu_idx;
            if(to_gpu_idx >= from_gpu_id) to_gpu_id ++;
            int global_remote_node_idx = tid % (num_out_nodes * (num_gpus - 1));
            int remote_node = out_nodes[tid % num_out_nodes];
            if(global_remote_node_idx == 0){
		//info[from_gpu_id].indptr_R.data[0] = 0;
            }
            if(is_selected(index_out_nodes_remote, tid)){
                auto write_index = index_out_nodes_remote[tid];
                info[from_gpu_id].out_nodes_remote.add_position_offset(remote_node, write_index );
                auto remote_pos = write_index - info[from_gpu_id].out_nodes_remote.offset;
                info[from_gpu_id].indptr_R.add_value_offset(index_indptr_remote[tid], remote_pos);
                if(remote_pos == 1){
			info[from_gpu_id].indptr_R.data[0] = 0;
		}
		info[to_gpu_id].push_from_ids[from_gpu_id].add_position_offset(remote_node, write_index);
            }
            start += BLOCK_SIZE;
      }
            tileId += gridDim.x;
    }
}

template<int BLOCKSIZE, int TILESIZE>
__global__ void fill_in_nodes(long * index_in_nodes, \
    long * index_out_nodes_local, \
    PushSlicer::LocalGraphInfo *info, int num_gpus,\
      long * in_nodes, size_t num_in_nodes,
      size_t num_out_nodes){
        int tileId = blockIdx.x;
        int lastTile = ( num_in_nodes * num_gpus- 1)/TILE_SIZE + 1;
        while(tileId < lastTile){
        int start = threadIdx.x + (tileId * TILE_SIZE);
        int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), num_in_nodes * num_gpus);
        while(start < end){
          int tid = start;
          int gpu_id = tid/ (num_in_nodes);
          auto in_node_idx = tid % num_in_nodes;
          if(in_node_idx < num_out_nodes){
            if(is_selected(index_out_nodes_local, (num_out_nodes * gpu_id + in_node_idx))){
                long in_node = in_nodes[tid % num_in_nodes];
                auto d = info[gpu_id].out_nodes_local;
                auto write_index = index_out_nodes_local[num_out_nodes * gpu_id + in_node_idx] - d.offset - 1;
                info[gpu_id].in_nodes.data[write_index] = in_node;
                start += BLOCK_SIZE;
                continue;
            }
          }
          if(is_selected(index_in_nodes, tid)){
            long in_node = in_nodes[tid % num_in_nodes];
            auto write_index = index_in_nodes[tid] + info[gpu_id].num_out_local;
            info[gpu_id].in_nodes.add_position_offset(in_node, write_index);
          }
          start += BLOCK_SIZE;
        }
        tileId += gridDim.x;
        }

}



void PushSlicer::resize_bipartite_graphs(PartitionedLayer &ps,int num_in_nodes,
    int num_out_nodes, int num_edges){
    transform::self_inclusive_scan(ps.index_in_nodes);
    transform::self_inclusive_scan(ps.index_out_nodes_local);
    transform::self_inclusive_scan(ps.index_out_nodes_remote);
    transform::self_inclusive_scan(ps.index_indptr_local);
    transform::self_inclusive_scan(ps.index_indptr_remote);
    transform::self_inclusive_scan(ps.index_edge_local);
    transform::self_inclusive_scan(ps.index_edge_remote);
    size_t size; size_t offset;
    for(int i =0; i < this->num_gpus; i++){
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
      bp.indptr_L.resize(0);
      if(size != 0)bp.indptr_L.resize(size + 1);
      bp.out_degree_local.resize(size);
      info.indptr_L.data = bp.indptr_L.ptr();
      info.indptr_L.offset = offset_indptr;
      info.out_nodes_local.data = bp.out_nodes_local.ptr();
      info.out_nodes_local.offset = offset;
      info.out_degree_local.data = bp.out_degree_local.ptr();
      info.out_degree_local.offset = offset;
      bp.self_ids_offset = size;
      info.num_out_local = size;

      // In nodes
      size = ps.index_in_nodes[num_in_nodes * (i + 1) - 1];
      offset = 0;
      if(i != 0){
        offset = ps.index_in_nodes[num_in_nodes * i - 1];
      }
      size = size - offset;
      bp.in_nodes.resize(size + info.num_out_local );
      bp.num_in_nodes_local = size + info.num_out_local  ;
      info.in_nodes.data = bp.in_nodes.ptr();
      info.in_nodes.offset = offset;



      // Out Nodes Remote
      size = ps.index_out_nodes_remote[(num_out_nodes * (this->num_gpus - 1)) * (i + 1)- 1];
      offset = 0;
      offset_indptr = 0;
      if( i != 0){
        offset_indptr = ps.index_indptr_remote[(num_out_nodes * (this->num_gpus - 1)) * i - 1];
        offset = ps.index_out_nodes_remote[(num_out_nodes * (this->num_gpus - 1)) * i - 1];
      }

      size = size - offset;
      bp.indptr_R.resize(0);
      if(size != 0)bp.indptr_R.resize(size + 1);
      info.indptr_R.data = bp.indptr_R.ptr();
      info.indptr_R.offset = offset_indptr;
      bp.out_nodes_remote.resize(size);
      bp.num_out_remote = size;
      info.out_nodes_remote.data = bp.out_nodes_remote.ptr();
      info.out_nodes_remote.offset = offset;

      auto global_offset = 0;
      if(i != 0)global_offset = num_out_nodes * (this->num_gpus - 1) * i ;
      // 2 data structures to_offsets, push_to_ids must be populated.
      bp.to_offsets[0] = 0;
      for(int to = 0; to < this->num_gpus; to++){
        if(i == to) {bp.to_offsets[to + 1] = bp.to_offsets[to]; continue;}
        auto j = to;
        if(j > i) j --;
        long start = 0;
        if(!((j == 0) && (i == 0))){
          start = ps.index_out_nodes_remote[global_offset + (j  * num_out_nodes) - 1];
        }
        auto end = ps.index_out_nodes_remote[global_offset + ((j + 1)  * num_out_nodes) - 1];
        bp.to_offsets[to + 1] = end - start + bp.to_offsets[to];
        ps.bipartite[to]->push_from_ids[i].resize(end - start);
        this->host_graph_info[to].push_from_ids[i].data = ps.bipartite[to]->push_from_ids[i].ptr();
        this->host_graph_info[to].push_from_ids[i].offset = start;
      }
      // Edges  Local
      offset = 0;
      if(i !=0 ){
        offset = ps.index_edge_local[num_edges * i - 1];
      }
      auto end = ps.index_edge_local[num_edges * (i + 1) - 1];
      bp.indices_L.resize(end - offset);
      info.indices_L.data = bp.indices_L.ptr();
      info.indices_L.offset = offset;
      // Edges Remote

      offset = 0;
      if(i != 0){
        offset = ps.index_edge_remote[num_edges * (num_gpus - 1) * i -1];
      }
      size = ps.index_edge_remote[num_edges * (num_gpus - 1) * (i + 1) -1];
      bp.indices_R.resize(size  - offset);
      info.indices_R.data = bp.indices_R.ptr();
      info.indices_R.offset = offset;
    }
    copy_graph_info();
}

void PushSlicer::slice_layer(device_vector<long> &layer_nds,
      Block &bs, PartitionedLayer &ps, bool last_layer){

    // Stage 1 Edge Partitioning
    // Layer nodes must be broken into Local and remote and ordered by Destination . Hence squared
    // Edges for both nodes exist in one vector
    // Self nodes also exist in one vector
    size_t num_out_nodes = layer_nds.size();
    size_t num_in_nodes = bs.layer_nds.size();
    size_t num_edges = bs.indices.size();

    int G = this->num_gpus;
    ps.resize_selected_push(num_out_nodes * G, num_out_nodes * G * (G-1),\
             num_edges * G, num_edges * (G-1) * G, num_in_nodes * G);

    // gpuErrchk(cudaDeviceSynchronize());
    partition_edges_push<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(layer_nds.size()),BLOCK_SIZE>>>\
        (this->workload_map.ptr(),\
          layer_nds.ptr(), layer_nds.size(),\
          bs.layer_nds.ptr(), bs.layer_nds.size(),\
          bs.offsets.ptr(), bs.indices.ptr(), bs.indices.size(), \
          this->num_nodes,\
          ps.index_in_nodes.ptr(), ps.index_out_nodes_local.ptr(), ps.index_out_nodes_remote.ptr(),\
          ps.index_indptr_local.ptr(), ps.index_indptr_remote.ptr(),\
          ps.index_edge_local.ptr(), ps.index_edge_remote.ptr(),\
          last_layer, this->storage_map_flattened,this->num_gpus);
    #ifdef DEBUG

    #endif
    gpuErrchk(cudaDeviceSynchronize());
    // Stage 2 get sizes of Offsets for all graphs
    // Inclusive Scan Everything
    resize_bipartite_graphs(ps, num_in_nodes, num_out_nodes, num_edges);
    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif

    // ps.debug_index();
    // Fill in local Node Info

    // Fill in Local Edges

    // Fill in Remote Edges
    fill_out_nodes_remote<BLOCK_SIZE, TILE_SIZE>\
      <<<GRID_SIZE(ps.index_out_nodes_remote.size()), TILE_SIZE>>>\
        (ps.index_out_nodes_remote.ptr(), \
            ps.index_out_nodes_remote.size(), \
              ps.index_indptr_remote.ptr(), \
                this->device_graph_info, num_gpus,\ // Meta data
                  layer_nds.ptr(), layer_nds.size());
    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif

    // Fill Local Node Info
    fill_out_nodes_local<BLOCK_SIZE, TILE_SIZE>\
    <<<GRID_SIZE( ps.index_out_nodes_local.size()),BLOCK_SIZE>>>\
        (ps.index_out_nodes_local.ptr(), ps.index_out_nodes_local.size(), \
            ps.index_indptr_local.ptr(), \
              this->device_graph_info, this->num_gpus,\ // Meta data
              layer_nds.ptr(), bs.in_degree.ptr(), layer_nds.size());
              #ifdef DEBUG
                gpuErrchk(cudaDeviceSynchronize());
              #endif

    fill_indices_remote<BLOCK_SIZE, TILE_SIZE>\
    <<<GRID_SIZE( ps.index_edge_remote.size()),BLOCK_SIZE>>>(ps.index_edge_remote.ptr(), num_edges, \
          ps.index_in_nodes.ptr(),  num_in_nodes,\
            ps.index_out_nodes_local.ptr(), num_out_nodes, \
            this->device_graph_info, num_gpus, bs.indices.ptr());
            #ifdef DEBUG
              gpuErrchk(cudaDeviceSynchronize());
            #endif

    fill_indices_local<BLOCK_SIZE, TILE_SIZE>\
    <<<GRID_SIZE( ps.index_edge_local.size()),BLOCK_SIZE>>>(bs.indices.ptr(),
         ps.index_edge_local.ptr(), num_edges, \
         ps.index_in_nodes.ptr() , num_in_nodes, \
         ps.index_out_nodes_local.ptr(), num_out_nodes, \
          this->device_graph_info, num_gpus);
         #ifdef DEBUG
           gpuErrchk(cudaDeviceSynchronize());
         #endif

  fill_in_nodes<BLOCK_SIZE, TILE_SIZE>\
        <<<GRID_SIZE( ps.index_in_nodes.size()),BLOCK_SIZE>>>(ps.index_in_nodes.ptr(), \
        ps.index_out_nodes_local.ptr(), \
       this->device_graph_info, num_gpus, bs.layer_nds.ptr(), num_in_nodes, num_out_nodes);

    #ifdef DEBUG
      gpuErrchk(cudaDeviceSynchronize());
    #endif

    // Fill Remote Node Info

}
