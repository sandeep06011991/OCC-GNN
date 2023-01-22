#pragma once
#include <vector>
#include <cassert>
#include "bipartite.h"
#include <thrust/device_vector.h>
#include "util/cuda_utils.h"

class PartitionedLayer{
  public:
    BiPartite* bipartite[4];
    // Contains expected size of local graphs
    thrust::device_vector<long> index_offset_map[16];
    thrust::device_vector<long> index_indices_map[16];
    // Used to capture expected number of nodes and edges in all graphs
    void * device_offset_map;
    void * device_indices_map;

    // There are 16 local graphs in total
    // num_gpus * num_gpus to represent for and to nodes.
    void * device_local_indptr_map;
    void * device_local_indices_map;
    //  Contains to nds
    void * device_local_to_nds_map;
    void * device_out_nodes_degree_map;

    PartitionedLayer(){
      // this->bipartite = (BiPartite **)malloc(sizeof(BiPartite *) * 4);
      for(int i=0;i<4;i++){
        this->bipartite[i] = new BiPartite(i);
      }
      gpuErrchk(cudaMalloc(&device_offset_map, sizeof(long *) * 16));
      gpuErrchk(cudaMalloc(&device_indices_map, sizeof(long *) * 16));
      gpuErrchk(cudaMalloc(&device_local_indptr_map, sizeof(long *)  * 16));
      gpuErrchk(cudaMalloc(&device_local_indices_map, sizeof(long *)  * 16));
      gpuErrchk(cudaMalloc(&device_local_to_nds_map, sizeof(long *)  * 16));
      gpuErrchk(cudaMalloc(&device_out_nodes_degree_map, sizeof(long *)  * 4));
    }

    void resize_index_and_offset_map(int indptr_size, int indices_size){
       void * local_offset[16];
       void * local_indices[16];
       std::cout << "resizing number of hnodes ot "<< indptr_size <<" ";
       for(int i = 0; i < 16; i++){
         index_offset_map[i].resize(indptr_size);
         local_offset[i] = thrust::raw_pointer_cast(index_offset_map[i].data());
         index_indices_map[i].resize(indices_size);
         local_indices[i] = thrust::raw_pointer_cast(index_indices_map[i].data());
       }
       gpuErrchk(cudaMemcpy(device_offset_map, local_offset, 16 * sizeof (void *), cudaMemcpyHostToDevice));
       gpuErrchk(cudaMemcpy(device_indices_map, local_indices, 16 * sizeof (void *), cudaMemcpyHostToDevice));
    }


  void resize_local_graphs(long * local_graph_nodes,long * local_graph_edges){
    void * local_offset[16];
    void * local_indices[16];
    void * local_to_nds[16];
    void * out_nodes_degree[4];
    for(int i=0;i < 4; i++){
        for(int j = 0; j <4;j++){
          if (i == j){
            bipartite[i]->out_degree_local.resize(local_graph_nodes[i * 4 + j]);
            out_nodes_degree[i] = thrust::raw_pointer_cast(bipartite[i]->out_degree_local.data());
          }
          bipartite[i]->indptr_[j].resize(local_graph_nodes[i * 4 + j] + 1);
          std::cout << "Size to ids" << local_graph_nodes[i * 4 + j] << " \n";
          bipartite[i]->to_ids_[j].resize(local_graph_nodes[i * 4 + j]);
          local_to_nds[i * 4 + j] = thrust::raw_pointer_cast(bipartite[i]->to_ids_[j].data());
          local_offset[i * 4 + j] = thrust::raw_pointer_cast(bipartite[i]->indptr_[j].data());
          bipartite[i]->indices_[j].resize(local_graph_edges[i * 4 + j] + 1);
          local_indices[i * 4 + j] = thrust::raw_pointer_cast(bipartite[i]->indices_[j].data());
        }
      }
      gpuErrchk(cudaMemcpy(device_local_indptr_map, local_offset, 16 * sizeof (void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_local_indices_map, local_indices, 16 * sizeof (void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_local_to_nds_map, local_to_nds, 16 * sizeof (void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_out_nodes_degree_map, out_nodes_degree, 4 * sizeof(void *), cudaMemcpyHostToDevice))
    }

    void clear(){
      for(int i=0;i<4;i++){
        this->bipartite[i]->refresh();
      }
    }

    void debug(){
        for(int i=0;i<4;i++){
          std::cout << "Bipartite graph" << i << "\n";
          bipartite[i]->debug();
        }
    }
    ~PartitionedLayer(){
       for(int i=0;i<4;i++){
         delete this->bipartite[i];
       }
    }
};

class PartitionedSample{
public:
  int num_layers = 3;
  // Fixed max layers == 4
  // Can be made a compile time constant. Do later.
  // Can allocate max possible layers or use compiler directives.
  PartitionedLayer layers[4];

  // From ids are storage order ids in the local cache or local feature
  // To ids are the position they are moved to in the input tensor
  std::vector<long> cache_hit_from[4];
  std::vector<long> cache_hit_to[4];
  std::vector<long> cache_miss_from[4];
  std::vector<long> cache_miss_to[4];
  // Nodes of the final raining values.
  std::vector<long> last_layer_nodes[4];

  PartitionedSample(int num_layers){
    this->num_layers = num_layers;
  }

  void clear(){
    for(int i=0;i<num_layers;i++){
      layers[i].clear();
    }
    for(int i=0;i<4;i++){
      cache_hit_from[i].clear();
      cache_hit_to[i].clear();
      cache_miss_from[i].clear();
      cache_miss_to[i].clear();
    }
  }

  void debug(){
    for(int i=0;i < num_layers; i++){
      std::cout << "Layer" << i <<"\n";
      layers[i].debug();

    }
  }
};
