#include "bipartite.h"
#include "util/cuda_utils.h"
// Replace everything with one local ordering.
void BiPartite::reorder_local(DuplicateRemover *dr){
  dr->clear();
  remove_duplicates(push_to_ids_[gpu_id]);
  dr->order(push_to_ids_[gpu_id]);
  // Fixes to match with cslicer begin.
  self_in_nodes = push_to_ids_[gpu_id];
  // Fixes end to match with cslicer
  int c1 = push_to_ids_[gpu_id].size();
  num_out_local = push_to_ids_[gpu_id].size();
  // It is a seperate graph.
  out_nodes_local = push_to_ids_[gpu_id];
  dr->clear();
  to_offsets[0] = 0;
  for(int i=0;i<num_gpus;i++){
  if(i != gpu_id){
      remove_duplicates(push_to_ids_[gpu_id]);
      dr->order(push_to_ids_[gpu_id]);
      cudaDeviceSynchronize();
      to_offsets[i + 1] = to_offsets[i] +  push_to_ids_[i].size();
      out_nodes_remote.insert(out_nodes_remote.end(), push_to_ids_[i].begin(), push_to_ids_[i].end());
      push_to_ids_[i].clear();
    }else{
    	to_offsets[i+1] = to_offsets[i];
    }
  }

  num_out_remote = to_offsets[num_gpus];
  dr->clear();

  // Order src nodes and associated data structures.
  // self nodes;
  dr->order(self_in_nodes);
  self_ids_offset = self_in_nodes.size();
  in_nodes.clear();
  // Code change to match  begin
  for(int i=0;i < num_gpus;i ++){
    in_nodes.insert(in_nodes.end(), indices_[i].begin(), indices_[i].end());
  }
  // Code change to match  end
  remove_duplicates(in_nodes);
  dr->remove_nodes_seen(in_nodes);
  dr->order(in_nodes);

  in_nodes_.insert(in_nodes_.end(), self_in_nodes.begin(), self_in_nodes.end());
  in_nodes_.insert(in_nodes_.end(), in_nodes.begin(), in_nodes.end());
  in_nodes.clear();
  in_nodes = in_nodes_;
  assert(c1 == self_ids_offset);
  // local_id
  num_in_nodes_local = in_nodes.size();
  // Why is this zero
  // point is to index into pull -in nodes
  pull_from_offsets[0] = 0;
  for(int i=0;i<num_gpus; i++){
    if(i!=gpu_id){
      // Minor FIX. remove duplicates from pull_from_ids
      remove_duplicates(pull_from_ids_[i]);
      dr->order(pull_from_ids_[i]);
      pulled_in_nodes.insert(pulled_in_nodes.end(),pull_from_ids_[i].begin(), pull_from_ids_[i].end());
      pull_from_offsets[i + 1] = pull_from_offsets[i] + pull_from_ids_[i].size();
      pull_from_ids_[i].clear();
    }else{
      pull_from_offsets[i + 1] = pull_from_offsets[i];
    }
  }
  num_in_nodes_pulled = pulled_in_nodes.size();
  // Create all local graphs.
  indptr_L = indptr_[gpu_id];
  indices_L = indices_[gpu_id];
  dr->replace(indices_L);
  // Remote partition nodes
  for(int i=0;  i<num_gpus;   i++){
      if((i!=gpu_id) && (indptr_[i].size()!=0)){
          if(indptr_R.size() != 0) {
             int back = indptr_R.back();
             for(int j=0;j<indptr_[i].size();j++){
                indptr_[i][j] += back;
             }
             indptr_R.resize(indptr_R.size()-1);
          }
          dr->replace(indices_[i]);
          indptr_R.insert(indptr_R.end(),indptr_[i].begin(), indptr_[i].end());
          indices_R.insert(indices_R.end(), indices_[i].begin(), indices_[i].end());
      }
  }
  dr->clear();
}
