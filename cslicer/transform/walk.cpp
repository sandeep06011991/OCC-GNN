#include "graph/sample.h"
#include <vector>
#include <iostream>
using namespace std;

#define char MASK;

struct gpu_meta{
  int set_partitions;
  int partitions[4];
};

gpu_meta meta_dict[16];

// Populate the gpu_meta flag.
void populate_meta_dict(){
  for(int i=0;i < 16; i++){
    meta_dict[i].set_partitions = 0;
    for(int gpu_id=0;gpu_id< 4;gpu_id ++ ){
      if(i & (1<<gpu_id)){
        int curr_partition = meta_dict[i].set_partitions;
        meta_dict[i].partitions[curr_partition] = gpu_id;
        meta_dict[i].set_partitions ++ ;
      }
    }
  }
}
// Walk and check the constructed sample.
// For each node in the internal layer.
// populate a mask of partitions that need it and a mask of partitions that can compute it.
// The assigned partition must be one of the intersection of the 2.
// If it is not simply switching to it will decrease shuffle cost.
vector<int> color_with_dest_gpu(vector<long> &offsets, vector<long> &indices,
                        int *workload_map){
  vector<int> partition_color;
  for(int i=0;i< offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int color = 0;
    for(; start < end; start ++){
       long dest = indices[start];
        color = color | (1 << workload_map[dest]);
    }
    partition_color.push_back(color);
  }
  return partition_color;
}


vector<int> color_with_src_gpu(vector<long>& src_nodes,
                      vector<long> &offsets, vector<long> &indices,
                        int *workload_map, vector<long> &layer_nds, int num_nodes){
  int * map = (int *)malloc(sizeof(int) * num_nodes);
  memset(map, 0, sizeof(int) * num_nodes);
  for(int i=0;i< offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int color = 0;
    long src = src_nodes[i];
    for(; start < end; start ++){
     long dest = indices[start];
          assert(workload_map[src] >=0 && workload_map[src] < 4);
        map[dest] |=  (1 << workload_map[src]);

    }
  }
  vector<int> partition_color;
  for(int i=0;i < layer_nds.size(); i++){
    partition_color.push_back(map[layer_nds[i]]);
  }
  free(map);
  return partition_color;
}

void check_allocation_for_optimality(Sample &s, int *workload_map, long num_nodes){
  vector<int> partition_map[s.num_layers + 1];

  for(int i=1; i < s.num_layers; i ++){
    int correct_nodes = 0;
    int bad_nodes = 0;

      vector<int> in_partitions = color_with_dest_gpu(s.block[i+1]->offsets,
              s.block[i+1]->indices, workload_map);
      vector<int> out_partitions = color_with_src_gpu(s.block[i-1]->layer_nds, s.block[i]->offsets,
              s.block[i]->indices, workload_map,s.block[i]->layer_nds, num_nodes);
      assert(in_partitions.size() == out_partitions.size());
      for(int j=0; j < s.block[i]->layer_nds.size(); j++){
        long nd =  s.block[i]->layer_nds[j];
        if(!((1 << workload_map[nd]) & in_partitions[j] & out_partitions[j])){
          // std::cout << "Missmatch " << (1 << workload_map[nd]) << " " << in_partitions[j] << " " <<  out_partitions[j] <<"\n";
          bad_nodes ++ ;
        }else{
          if(1<<workload_map[nd] == in_partitions[j] && (in_partitions[j] == out_partitions[j])){

          }else{
            correct_nodes ++;

          }
          // std::cout << "good partition\n";
        }
      }
      // 10% overall decrease in traffic if I have smarter coloring. 
      std::cout << "layer" << i << " " << correct_nodes <<" " << bad_nodes << " " << s.block[i]->layer_nds.size() << "\n";
    }

}
