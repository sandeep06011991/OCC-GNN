#pragma once
#include <vector>
#include <cassert>
#include "bipartite.h"
#include <thrust/device_vector.h>
#include "util/cuda_utils.h"

class PartitionedLayer{
  public:
    BiPartite* bipartite[8];
    // Contains expected size of local graphs
    // map[nd1, nd2] -> refers to nd1 from nd2
    // dest node is nd1 and src node is nd2
    thrust::device_vector<long> index_offset_map[8 * 8];
    thrust::device_vector<long> index_indices_map[8 * 8];
    // Used to capture expected number of nodes and edges in all graphs
    void * device_offset_map;
    void * device_indices_map;

    // There are 16 local graphs in total
    // num_gpus * num_gpus to represent for and to nodes.
    void * device_local_indptr_map;
    void * device_local_indices_map;
    //  Contains to nds
    void * device_local_to_nds_map;
    void * device_local_from_nds_map;
    void * device_out_nodes_degree_map;
    int num_gpus = -1;
    PartitionedLayer(){}

    void set_number_of_gpus(int num_gpus){
      this->num_gpus = num_gpus;
      // this->bipartite = (BiPartite **)malloc(sizeof(BiPartite *) * 4);
      for(int i=0; i<num_gpus; i++){
        this->bipartite[i] = new BiPartite(i);
      }
      int map_size = this->num_gpus * this->num_gpus;
      gpuErrchk(cudaMalloc(&device_offset_map, sizeof(long *) * map_size ));
      gpuErrchk(cudaMalloc(&device_indices_map, sizeof(long *) * map_size ));
      gpuErrchk(cudaMalloc(&device_local_indptr_map, sizeof(long *)  * map_size ));
      gpuErrchk(cudaMalloc(&device_local_indices_map, sizeof(long *)  * map_size ));
      gpuErrchk(cudaMalloc(&device_local_to_nds_map, sizeof(long *)  * map_size ));
      gpuErrchk(cudaMalloc(&device_local_from_nds_map, sizeof(long *)  * map_size ));
      gpuErrchk(cudaMalloc(&device_out_nodes_degree_map, sizeof(long *)  * this->num_gpus));
    }

    void resize_index_and_offset_map(int indptr_size, int indices_size){
       int N = this->num_gpus * this->num_gpus;
       // n refers to number of graph partitions
       void * local_offset[N];
       void * local_indices[N];
       for(int i = 0; i < N; i++){
         index_offset_map[i].resize(indptr_size);
         thrust::fill(index_offset_map[i].begin(), index_offset_map[i].end(), 0);
         local_offset[i] = thrust::raw_pointer_cast(index_offset_map[i].data());
         index_indices_map[i].resize(indices_size);
         thrust::fill(index_indices_map[i].begin(), index_indices_map[i].end(), 0);

         local_indices[i] = thrust::raw_pointer_cast(index_indices_map[i].data());
       }
       gpuErrchk(cudaMemcpy(device_offset_map, local_offset, N * sizeof (void *), cudaMemcpyHostToDevice));
       gpuErrchk(cudaMemcpy(device_indices_map, local_indices, N * sizeof (void *), cudaMemcpyHostToDevice));
    }

  void debug_partition_edges(){
      for(int i=0;i <this->num_gpus; i++){
        for(int j=0; j < this->num_gpus; j++){
          std::cout << "Dest" << i << ":src" << j <<"\n";
          debugVector(index_offset_map[i * this->num_gpus + j], "Indptr");
          debugVector(index_indices_map[i * this->num_gpus + j], "Indices");
        }
      }

  }
  void resize_local_graphs(long * local_graph_nodes,long * local_graph_edges){
    int N = this->num_gpus * this->num_gpus;
    void * local_offset[N];
    void * local_indices[N];
    void * local_to_nds[N];
    void * local_from_nds[N];
    void * out_nodes_degree[this->num_gpus];
    for(int dest =0;dest < this->num_gpus; dest++){
      for(int src = 0; src < this->num_gpus ;src++){
          std::cout << "Resize " << local_graph_nodes[i*4 + j] << " " << local_graph_edges[i * 4 + j] <<"\n";
          if (src == dest){
            bipartite[dest]->out_degree_local.resize(local_graph_nodes[src * this->num_gpus + dest]);
            out_nodes_degree[dest] = thrust::raw_pointer_cast(bipartite[dest]->out_degree_local.data());
          }
          bipartite[src]->indptr_[dest].resize(local_graph_nodes[dest * this->num_gpus + src] + 1);
          std::cout << "Size to ids" << local_graph_nodes[dest * this->num_gpus + src] << " \n";
          bipartite[src]->to_ids_[dest].resize(local_graph_nodes[dest * this->num_gpus + src]);
          local_to_nds[dest * this->num_gpus + src] = thrust::raw_pointer_cast(bipartite[src]->to_ids_[dest].data());
          local_offset[dest * this->num_gpus + j] = thrust::raw_pointer_cast(bipartite[src]->indptr_[dest].data());
          bipartite[src]->indices_[dest].resize(local_graph_edges[dest * this->num_gpus + src]);
          local_indices[dest * this->num_gpus + src] = thrust::raw_pointer_cast(bipartite[src]->indices_[dest].data());
          bipartite[dest]->from_ids[src].resize(local_graph_edges[dest * this->num_gpus + src]);
          local_from_nds[dest * this->num_gpus + src] = thrust::raw_pointer_cast(bipartite[dest]->from_ids[src].data());
        }
      }
      gpuErrchk(cudaMemcpy(device_local_indptr_map, local_offset, N * sizeof (void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_local_indices_map, local_indices, N * sizeof (void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_local_to_nds_map, local_to_nds, N * sizeof (void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_local_from_nds_map, local_from_nds, N * sizeof(void *), cudaMemcpyHostToDevice));
      gpuErrchk(cudaMemcpy(device_out_nodes_degree_map, out_nodes_degree, this->num_gpus * sizeof(void *), cudaMemcpyHostToDevice))
    }

    void clear(){
      for(int i=0;i<this->num_gpus;i++){
        this->bipartite[i]->refresh();
      }
    }

    void debug(){
        for(int i=0;i<this->num_gpus;i++){
          std::cout << "Bipartite graph" << i << "\n";
          bipartite[i]->debug();
        }
    }
    ~PartitionedLayer(){
       for(int i=0;i<this->num_gpus;i++){
         delete this->bipartite[i];
       }
    }
};

class PartitionedSample{
public:
  int num_layers = MAX_LAYERS;
  // Fixed max layers == 4
  // Can be made a compile time constant. Do later.
  // Can allocate max possible layers or use compiler directives.
  PartitionedLayer layers[MAX_DEVICES];

  // From ids are storage order ids in the local cache or local feature
  // To ids are the position they are moved to in the input tensor
  thrust::device_vector<long> cache_hit_from[MAX_DEVICES];
  thrust::device_vector<long> cache_hit_to[MAX_DEVICES];
  thrust::device_vector<long> cache_miss_from[MAX_DEVICES];
  thrust::device_vector<long> cache_miss_to[MAX_DEVICES];
  // Nodes of the final raining values.
  thrust::device_vector<long> last_layer_nodes[MAX_DEVICES];
  int num_gpus = -1;

  PartitionedSample(int num_layers, int num_gpus){
    this->num_layers = num_layers;
    this->num_gpus = num_gpus;
    for(int i=0;i < this->num_layers; i ++){
      layers[i].set_number_of_gpus(this->num_gpus);
    }

  }

  void clear(){
    for(int i=0;i<num_layers;i++){
      layers[i].clear();
    }
    for(int i=0;i<num_gpus;i++){
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
