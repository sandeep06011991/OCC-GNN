#include "util/duplicate.h"
#include <vector>
#include <iostream>
using namespace std;
#pragma once

class BiPartite{
public:
  vector<long> in_nodes;

  vector<long> indptr;

  vector<long> out_nodes;
  vector<long> owned_out_nodes;

  // Filled afer reordering
  vector<long> indices;
  // Easy fill
  vector<long> from_ids[4];
  // Refill.
  vector<long> to_ids[4];

  vector<long> self_ids_in;
  vector<long> self_ids_out;

  int gpu_id = -1;


  BiPartite(int gpu_id){
    this->gpu_id = gpu_id;
  }

  inline void add_self_edge(long nd1){
    if((self_ids_in.size()!=0) && (self_ids_in.back() == nd1)){
      return;
    }
    self_ids_in.push_back(nd1);
    self_ids_out.push_back(nd1);
  }

  inline void add_from_node(long nd1, int gpu_id){
      if((from_ids[gpu_id].size() != 0 ) && (from_ids[gpu_id].back() == nd1)){
        return;
      }
      from_ids[gpu_id].push_back(nd1);
  }

  inline void add_to_node(long nd1, int gpu_id){
    if((to_ids[gpu_id].size() != 0 ) && (to_ids[gpu_id].back() == nd1)){
      return;
    }
    to_ids[gpu_id].push_back(nd1);
  }

  inline void add_edge(int nd1, int nd2, bool islocal){
      if(islocal && ((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
        owned_out_nodes.push_back(nd1);
      }
      if(out_nodes.size() == 0 || out_nodes.back() != nd1){
          out_nodes.push_back(nd1);
          indptr.push_back(1);
      }
      in_nodes.push_back(nd2);
      int l = indptr.size();
      indptr[l] = indptr[l] + 1;
  }

  void refresh(){
    for(int i=0;i<4;i++){
      from_ids[i].clear();
      to_ids[i].clear();
    }
    indptr.clear();
    indices.clear();
    self_ids_in.clear();
    self_ids_out.clear();

    in_nodes.clear();
    out_nodes.clear();
    owned_out_nodes.clear();
  }

  void reorder(DuplicateRemover* dr);
};
