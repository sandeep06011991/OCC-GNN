#include "graph/sample.h"
#include <vector>
#include <iostream>
#include "transform/slice.h"
using namespace std;

#define char MASK;



gpu_meta meta_dict[16];

// Populate the gpu_meta flag.
void populate_meta_dict(){
  for(int i=0;i < 16; i++){
    meta_dict[i].set_partitions = 0;
    for(int gpu_id=0;gpu_id< 4;gpu_id ++ ){
      if((i & (1<<gpu_id)) != 0){
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
void  color_with_src_gpu(vector<long>& layer_nds, vector<long> &offsets, vector<long> &indices,
                        vector<int> & color_in ,   vector<int> &color_out, vector<long> &layer_in_nodes){
  for(int i=0;i< (int)offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int color = 0;
    long src = layer_nds[i];
    for(; start < end; start ++){
       long dest = layer_in_nodes[indices[start]];
       color_out[dest] = color_out[dest]  | color_in[src];
      }
  }
}




// Assume 3 hops or 4 layers
redundant  print_statistics(Sample &s,std::vector<int> ** layer_color, long num_nodes,
        vector<int>& workload_map, std::vector<int> storage_map[4]){
  // color first layer
  populate_meta_dict();
  for(auto nd:  s.block[0]->layer_nds){
      assert(workload_map[nd] <4);
      ( * layer_color[0])[nd] = 1 << (workload_map[nd]);
  }
  for(int i = 1; i < s.num_layers + 1; i ++){
    vector<int>& src_color = *layer_color[i-1];
    vector<int>& dest_color = *layer_color[i];
    color_with_src_gpu(s.block[i-1]->layer_nds, s.block[i]->offsets, s.block[i]->indices, src_color, dest_color,
          s.block[i]->layer_nds);
  }

  // Resetting
  int total_computation = 0;
  int redundant_computation = 0;
  int total_communication = 0;
  int redundant_communication = 0;
  for(int i=0;i<4;i++){
    for(int j=0; j < s.block[i]->layer_nds.size(); j++){
      auto nd = s.block[i]->layer_nds[j];
      int color = (*layer_color[i])[nd];

      if (i!=3){
          int degree= s.block[i+1]->offsets[j+1] - s.block[i+1]->offsets[j];
          total_computation += meta_dict[color].set_partitions * degree;
          redundant_computation += (meta_dict[color].set_partitions -1) * degree;
      }else{
        if(storage_map[i][nd] == -1){
          total_communication += meta_dict[color].set_partitions;
          redundant_communication += (meta_dict[color].set_partitions -1);
        }
      }
      (*layer_color[i])[nd] = 0;
    }
  }
  redundant r;
  r.total_computation = total_computation;
  r.redundant_computation = redundant_computation;
  r.total_communication = total_communication;
  r.redundant_communication = redundant_communication;
  return r;
}
