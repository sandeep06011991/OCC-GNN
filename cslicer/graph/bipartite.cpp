#include "bipartite.h"

// Replace everything with one local ordering.
void BiPartite::reorder_local(DuplicateRemover *dr){
  // Order destination nodes;
  dr->clear();
  dr->order_and_remove_duplicates(to_ids_[gpu_id]);
  int c1 = to_ids_[gpu_id].size();
  num_out_local = to_ids_[gpu_id].size();
  dr->clear();
  to_offsets[0] = 0;
  for(int i=0;i<4;i++){
   if(i != gpu_id){
      dr->order_and_remove_duplicates(to_ids_[i]);
      to_offsets[i + 1] = to_offsets[i] +  to_ids_[i].size();
      out_nodes_remote.insert(out_nodes_remote.end(), to_ids_[i].begin(), to_ids_[i].end());
      to_ids_[i].clear();
    }else{
    	to_offsets[i+1] = to_offsets[i];
    }

  }
  num_out_remote = to_offsets[5];
  dr->clear();

  // Order src nodes and associated data structures.
  // self nodes;
  dr->order_and_remove_duplicates(in_nodes);
  self_ids_offset = in_nodes.size();
  assert(c1 == self_ids_offset);
  // local_id
  in_nodes.insert(in_nodes.end(), indices_[gpu_id].begin(), indices_[gpu_id].end());
  dr->order_and_remove_duplicates(in_nodes);
  num_in_nodes_local = in_nodes.size();

  pull_from_offsets[0] = 0;
  for(int i=0;i<4; i++){
    if(i!=gpu_id){
      // Minor FIX. remove duplicates from pull_from_ids
      pulled_in_nodes.insert(pulled_in_nodes.end(),pull_from_ids_[i].begin(), pull_from_ids_[i].end());
      dr->order_and_remove_duplicates(pulled_in_nodes);
      pull_from_offsets[i + 1] = pulled_in_nodes.size();
      pull_from_ids_[i].clear();
    }else{
      pull_from_offsets[i + 1] = pull_from_offsets[i];
    }
  }

  // Create all local graphs.
  indptr_L = indptr_[gpu_id];
  indices_L = indices_[gpu_id];
  dr->replace(indices_L);

  // Remote partition nodes
  for(int i=0;i<4;i++){
      if(i!=gpu_id){
          if(indptr_R.size() != 0){
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

// void BiPartite::reorder(DuplicateRemover* dr){
//   dr->order_and_remove_duplicates(in_nodes);
//
//   num_in_nodes = dr->used_nodes.size();
//   dr->replace(indices);
//   dr->replace(self_ids_in);
//   dr->clear();
//
//   dr->order_and_remove_duplicates(out_nodes);
//   //dr->order_and_remove_duplicates(owned_out_nodes);
//
//   num_out_nodes = dr->used_nodes.size();
//   dr->replace(expand_indptr);
//   dr->replace(owned_out_nodes);
//   dr->replace(self_ids_out);
//   for(int i=0;i<4;i++){
//    // std::cout << "replace from and to ids " << from_ids[i].size() << " " << to_ids[i].size() <<"\n";
//     dr->replace(from_ids[i]);
//     dr->replace(to_ids[i]);
//   }
//   dr->clear();
// }
//
// void BiPartite::reorder_lastlayer(DuplicateRemover *dr, vector<int>& gpu_order, int gpu_capacity){
//   dr->order_and_remove_duplicates(in_nodes);
//
//   // order and remove used only to remove duplicates
//   // This dr object is not used to replace
//   // The original indices are used to replace
//   for(int nd: in_nodes){
//     if(gpu_order[nd] == -1){
//       missing_node_ids.push_back(nd);
//     }else{
//       cached_node_ids.push_back(nd);
//     }
//   }
//   dr->clear();
//   dr->order_and_remove_duplicates(cached_node_ids);
//   dr->order_and_remove_duplicates(missing_node_ids);
//   // In global order cached nodes will be put in first.
//   // for(int nd :in_nodes){
//   //    if(gpu_order[nd] == -1){
//   //      //std::cout << "bipartite reorder " << nd << " " << gpu_capacity <<"\n";
//   //      gpu_order[nd] = gpu_capacity;
//   //      gpu_capacity ++;
//   //      missing_node_ids.push_back(nd);
//   //    }
//   // }
//   // for(int i=0;i<indices.size();i++){
//   //   long nd = indices[i];
//   //   indices[i] = gpu_order[nd];
//   //   assert(gpu_order[nd] >= 0);
//   // }
//   // spdlog::info("Added missing nodes of size {}",missing_node_ids.size());
//   // std::cout << "\n";
//   // Possible
//   // assert(indices[indices.size()-1] > 0);
//   dr->replace(self_ids_in);
//   dr->replace(indices);
//   num_in_nodes = cached_node_ids.size() + missing_node_ids.size();
//
//   for(int i = 0;i < cached_node_ids.size(); i ++ ){
//     cached_node_ids[i] = gpu_order[cached_node_ids[i]];
//   }
//   // for(int i=0; i<self_ids_in.size(); i++){
//   //   long nd = self_ids_in[i];
//   //   self_ids_in[i] = gpu_order[nd];
//   //   assert(self_ids_in[i] >= 0);
//   // }
//   // num_in_nodes = gpu_capacity ;
//   // // + missing_node_ids.size();
//   // for(long nd:missing_node_ids){
//   //   gpu_order[nd] = -1;
//   // }
//   dr->clear();
//   dr->order_and_remove_duplicates(out_nodes);
//   //dr->order_and_remove_duplicates(owned_out_nodes);
//   num_out_nodes = dr->used_nodes.size();
//   //std::cout << "replace owned out nodes \n";
//   dr->replace(owned_out_nodes);
//   dr->replace(self_ids_out);
//   dr->replace(expand_indptr);
//   for(int i=0;i<4;i++){
//     //std::cout << "replace from and to ids " << i <<"\n";
//     dr->replace(from_ids[i]);
//     dr->replace(to_ids[i]);
//   }
//   dr->clear();
// }
