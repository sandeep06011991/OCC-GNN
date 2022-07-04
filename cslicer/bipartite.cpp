#include "bipartite.h"

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

  //std::cout << "last layer before reordering" << indices[indices.size()-1] <<"\n";
  for(int i=0;i<indices.size();i++){
    indices[i] = gpu_order[indices[i]];
    assert(indices[i] >=0);
  }
  assert(indices[indices.size()-1] > 0);
  //std::cout << "last layer" << indices[indices.size()-1] <<"\n";
  for(int i=0; i<self_ids_in.size(); i++){
    self_ids_in[i] = gpu_order[self_ids_in[i]];
    assert(self_indices_in[i] >=0);
  }
  num_in_nodes = gpu_capacity;


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
