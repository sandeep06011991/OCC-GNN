#include "util/dist_tensor.hh"
#include "gnn/dist_sage.hh"
#include <vector>


void DistSageAggr::forward(vector<int>& ind_ptr, vector<int>& indices,
          DistTensor& in, int num_nodes_out, int num_nodes_in){

      // // Assume a magic reordering map
      // for(int i=0; i<gpu_ids;i++){
      //   for(int j=0; j<gpu_ids;j++){
      //     ind_ptr = new Tensor<float>(ind_ptr);
      //     indices = new Tensor<float>(indices);
      //   }
      // }
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
