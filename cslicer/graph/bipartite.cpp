#include "bipartite.h"
#include "spdlog/spdlog.h"

void BiPartite::reorder(DuplicateRemover* dr){
  dr->order_and_remove_duplicates(in_nodes);
  num_in_nodes = dr->used_nodes.size();
  dr->replace(indices);
  dr->replace(self_ids_in);
  dr->clear();

  dr->order_and_remove_duplicates(out_nodes);
  //dr->order_and_remove_duplicates(owned_out_nodes);

  num_out_nodes = dr->used_nodes.size();
  dr->replace(expand_indptr);
  dr->replace(owned_out_nodes);
  dr->replace(self_ids_out);
  for(int i=0;i<4;i++){
   // std::cout << "replace from and to ids " << from_ids[i].size() << " " << to_ids[i].size() <<"\n";
    dr->replace(from_ids[i]);
    dr->replace(to_ids[i]);
  }
  dr->clear();
}

void BiPartite::reorder_lastlayer(DuplicateRemover *dr, vector<int>& gpu_order, int gpu_capacity){
  dr->order_and_remove_duplicates(in_nodes);
  // order and remove used only to remove duplicates
  // This dr object is not used to replace
  // The original indices are used to replace
  for(int nd :in_nodes){
     if(gpu_order[nd] == -1){
       gpu_order[nd] = gpu_capacity;
       gpu_capacity ++;
       missing_node_ids.push_back(nd);
     }
  }
  for(int i=0;i<indices.size();i++){
    long nd = indices[i];
    indices[i] = gpu_order[nd];
    assert(gpu_order[nd] >= 0);
  }
  // spdlog::info("Added missing nodes of size {}",missing_node_ids.size());
  // std::cout << "\n";
  // Possible
  // assert(indices[indices.size()-1] > 0);

  for(int i=0; i<self_ids_in.size(); i++){
    long nd = self_ids_in[i];
    self_ids_in[i] = gpu_order[nd];
    assert(self_ids_in[i] >= 0);
  }
  num_in_nodes = gpu_capacity ;
  // + missing_node_ids.size();
  for(long nd:missing_node_ids){
    gpu_order[nd] = -1;
  }
  dr->clear();
  dr->order_and_remove_duplicates(out_nodes);
  //dr->order_and_remove_duplicates(owned_out_nodes);
  num_out_nodes = dr->used_nodes.size();
  //std::cout << "replace owned out nodes \n";
  dr->replace(owned_out_nodes);
  dr->replace(self_ids_out);
  dr->replace(expand_indptr);
  for(int i=0;i<4;i++){
    //std::cout << "replace from and to ids " << i <<"\n";
    dr->replace(from_ids[i]);
    dr->replace(to_ids[i]);
  }
  dr->clear();
}
