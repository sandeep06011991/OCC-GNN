#include "bipartite.h"
#include "../util/cuda_utils.h"
#include "nvtx3/nvToolsExt.h"

// Replace everything with one local ordering.
void BiPartite::reorder_local(DuplicateRemover *dr){
  // nvtxRangePush("reorder");
  std::cout << "Not implemented\n";
  std::cout << "Ideally everything in this function is sorted, only accts to merge\n";
  // dr->clear();
  // remove_duplicates(push_to_ids_[gpu_id]);
  // dr->order(push_to_ids_[gpu_id]);
  // // Fixes to match with cslicer begin.
  // self_in_nodes = push_to_ids_[gpu_id];
  // // Fixes end to match with cslicer
  // int c1 = push_to_ids_[gpu_id].size();
  // num_out_local = push_to_ids_[gpu_id].size();
  // // It is a seperate graph.
  // out_nodes_local = push_to_ids_[gpu_id];
  // dr->clear();
  // to_offsets[0] = 0;
  // for(int i=0;i<num_gpus;i++){
  // if(i != gpu_id){
  //     remove_duplicates(push_to_ids_[gpu_id]);
  //     dr->order(push_to_ids_[gpu_id]);
  //     to_offsets[i + 1] = to_offsets[i] +  push_to_ids_[i].size();
  //     out_nodes_remote.insert(out_nodes_remote.end(), push_to_ids_[i].begin(), push_to_ids_[i].end());
  //     push_to_ids_[i].clear();
  //   }else{
  //   	to_offsets[i+1] = to_offsets[i];
  //   }
  // }
  // num_out_remote = to_offsets[num_gpus];
  // dr->clear();
  //
  // // Order src nodes and associated data structures.
  // // self nodes;
  // // Remove pull nodes
  // pull_from_offsets[0] = 0;
  // for(int i=0;i<num_gpus; i++){
  //   if(i!=gpu_id){
  //     // Minor FIX. remove duplicates from pull_from_ids
  //     remove_duplicates(pull_from_ids_[i]);
  //     dr->order(pull_from_ids_[i]);
  //     pulled_in_nodes.insert(pulled_in_nodes.end(),pull_from_ids_[i].begin(), pull_from_ids_[i].end());
  //     pull_from_offsets[i + 1] = pull_from_offsets[i] + pull_from_ids_[i].size();
  //     pull_from_ids_[i].clear();
  //   }else{
  //     pull_from_offsets[i + 1] = pull_from_offsets[i];
  //   }
  // }
  // // Calculate in_nodes not found
  // in_nodes.clear();
  // for(int i=0;i < num_gpus;i ++){
  //   in_nodes.insert(in_nodes.end(), indices_[i].begin(), indices_[i].end());
  //   remove_duplicates(pull_to_ids[i]);
  //   in_nodes.insert(in_nodes.end(), pull_to_ids[i].begin(), pull_to_ids[i].end());
  // }
  // dr->order(self_in_nodes);
  // remove_duplicates(in_nodes);
  // dr->remove_nodes_seen(in_nodes);
  //
  // dr->clear();
  // dr->order(self_in_nodes);
  // in_nodes_.clear();
  // in_nodes_.insert(in_nodes_.end(), self_in_nodes.begin(), self_in_nodes.end());
  // dr->order(in_nodes);
  // in_nodes_.insert(in_nodes_.end(), in_nodes.begin(), in_nodes.end());
  // num_in_nodes_local = in_nodes_.size();
  // in_nodes = in_nodes_;
  // dr->order(pulled_in_nodes);
  // // in_nodes_.insert(in_nodes_.end(), pulled_in_nodes.begin(), pulled_in_nodes.end());
  // self_ids_offset = self_in_nodes.size();
  //
  // assert(c1 == self_ids_offset);
  //
  // num_in_nodes_pulled = pulled_in_nodes.size();
  //
  //
  // // Create all local graphs.
  // indptr_L = indptr_[gpu_id];
  // indices_L = indices_[gpu_id];
  //
  // dr->replace(indices_L);
  // // Remote partition nodes
  // nvtxRangePush("increment");
  // for(int i=0;  i<num_gpus;   i++){
  //     if((i!=gpu_id) && (indptr_[i].size()!=0)){
  //         if(indptr_R.size() != 0) {
  //            long back = indptr_R.back();
  //            thrust::transform(thrust::device, indptr_[i].begin(), indptr_[i].end(),\
  //               indptr_[i].begin(),[=] __device__ (long n){return n + back;});
  //            // for(int j=0;j<indptr_[i].size();j++){
  //            //    indptr_[i][j] += back;
  //            // }
  //            indptr_R.resize(indptr_R.size()-1);
  //         }
  //         dr->replace(indices_[i]);
	//   // gpuErrchk(cudaDeviceSynchronize());
  //
	//   indptr_R.insert(indptr_R.end(),indptr_[i].begin(), indptr_[i].end());
  //         indices_R.insert(indices_R.end(), indices_[i].begin(), indices_[i].end());
  //     }
  // }
  // nvtxRangePop();
  // dr->clear();
  // nvtxRangePop();

}
