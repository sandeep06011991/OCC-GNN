#include "util/dist_tensor.hh"
#include "gnn/dist_sage.hh"
#include <vector>


void DistSageAggr::forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in){

      // some magic reordering map
      int * reorder_map = (int *)malloc(sizeof(int) * num_nodes_out);
      for(int i=0; i< num_nodes_out;i++){
        reorder_map[i] = i%2;
      }

      // Break 

      // // End: All code is positioned and memory is instantiated.
      // // Begin launch kernels.
      //
      //
      // for(int i=0; i < gpu_ids; i++){
      //   for(int j=0; j < gpu_ids; j++){
      //
      //   }
      // }
      //
      // // Start Reduction.
      // for(int i=0;i < gpu_ids; i++){
      //   for(int j=0; j <gpu_ids; j++){
      //
      //   }
      // }
      // //  Final result is this.
      // out = new DistTensor();

  }
