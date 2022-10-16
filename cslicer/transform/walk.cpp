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
                        vector<int> & color_in ,   vector<int> &color_out){
  for(int i=0;i< (int)offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int color = 0;
    long src = layer_nds[i];
    for(; start < end; start ++){
       long dest = indices[start];
       color_out[dest] = color_out[dest]  | color_in[src];
      }
  }
}




// Assume 3 hops or 4 layers
redundant  print_statistics(Sample &s,std::vector<int> ** layer_color, long num_nodes, vector<int>& workload_map){
  // color first layer
  for(auto nd:  s.block[0]->layer_nds){
      assert(workload[nd] <4);
      ( * layer_color[0])[nd] = 1 << (workload_map[nd]);
  }
  for(int i = 1; i < s.num_layers + 1; i ++){
    vector<int>& src_color = *layer_color[i-1];
    vector<int>& dest_color = *layer_color[i];
    color_with_src_gpu(s.block[i-1]->layer_nds, s.block[i]->offsets, s.block[i]->indices, src_color, dest_color );
  }

  // Resetting
  int total_computation = 0;
  int redundant_computation = 0;
  int total_communication = 0;
  int redundant_communication = 0;
  for(int i=0;i<4;i++){
    for(auto nd:s.block[i]->layer_nds){
      int color = (*layer_color[i])[nd];
      std::cout << color << "\n";
      if (i!=3){
          total_computation += meta_dict[color].set_partitions;
          redundant_computation += (meta_dict[color].set_partitions -1);
      }else{
        total_communication += meta_dict[color].set_partitions;
        redundant_communication += (meta_dict[color].set_partitions -1);
      }
      (*layer_color[i])[nd] = 0;
    }
  }
  std::cout << "total compuitation" << total_computation <<"\n";
  redundant r;
  r.total_computation = total_computation;
  r.redundant_computation = redundant_computation;
  r.total_communication = total_communication;
  r.redundant_communication = redundant_communication;
  return r;
}
//
// void check_allocation_for_optimality(Sample &s, int *workload_map, long num_nodes){
//   vector<int> partition_map[s.num_layers + 1];
//   for(int i=1; i < s.num_layers; i ++){
//     int correct_nodes = 0;
//     int bad_nodes = 0;
//
//       vector<int> in_partitions = color_with_dest_gpu(s.block[i+1]->offsets,
//               s.block[i+1]->indices, workload_map);
//       vector<int> out_partitions = color_with_src_gpu(s.block[i-1]->layer_nds, s.block[i]->offsets,
//               s.block[i]->indices, workload_map,s.block[i]->layer_nds, num_nodes);
//       assert(in_partitions.size() == out_partitions.size());
//       for(int j=0; j < s.block[i]->layer_nds.size(); j++){
//         long nd =  s.block[i]->layer_nds[j];
//         if(!((1 << workload_map[nd]) & in_partitions[j] & out_partitions[j])){
//           // std::cout << "Missmatch " << (1 << workload_map[nd]) << " " << in_partitions[j] << " " <<  out_partitions[j] <<"\n";
//           bad_nodes ++ ;
//         }else{
//           if(1<<workload_map[nd] == in_partitions[j] && (in_partitions[j] == out_partitions[j])){
//
//           }else{
//             correct_nodes ++;
//
//           }
//           // std::cout << "good partition\n";
//         }
//       }
//       // 10% overall decrease in traffic if I have smarter coloring.
//       std::cout << "layer" << i << " " << correct_nodes <<" " << bad_nodes << " " << s.block[i]->layer_nds.size() << "\n";
//     }
//
// }
