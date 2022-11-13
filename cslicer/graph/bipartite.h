#include "util/duplicate.h"
#include <vector>
#include <iostream>
using namespace std;
#pragma once

class BiPartite{

public:
  // sorted and no duplicate nodes using graph_ids.
  // Used to find mapping between local id and global id
  // nd2 is in_nodes as sampling is top down, but flow is bottom up.
  vector<long> in_nodes;
  vector<long> out_nodes;
  // Num in nodes can be much larger then in the actual graph, if we use
  // cached node order on the gpu.
  int num_in_nodes;
  int num_out_nodes;

  vector<long> indptr;
  vector<long> expand_indptr;
  vector<long> indices;
  vector<long> in_degree;

  // Used to fresh gpu map.
  vector<long> missing_node_ids;
  vector<long> cached_node_ids;
  vector<long> owned_out_nodes;

  // Filled afer reordering
  // Easy fill
  vector<long> from_ids[4];
  vector<long> to_ids[4];
  

  // Used for self attention.
  vector<long> self_ids_in;
  vector<long> self_ids_out;

  int gpu_id = -1;


  BiPartite(int gpu_id){
    this->gpu_id = gpu_id;
  }

  inline void add_self_edge(long nd1, int degree){
    if((self_ids_in.size()!=0) && (self_ids_in.back() == nd1)){
      return;
    }
    self_ids_in.push_back(nd1);
    self_ids_out.push_back(nd1);
    in_degree.push_back(degree);
    assert(degree > 0);
    // in_nodes and out nodes are pushed here instead of at add_edge
    // as self loops are not counted in the main graph
    //
    in_nodes.push_back(nd1);
    owned_out_nodes.push_back(nd1);
    if(indptr.size() == 0){
      indptr.push_back(0);
      indptr.push_back(0);
      if((out_nodes.size()==0) || (out_nodes.back() != nd1)){
        out_nodes.push_back(nd1);
      }
    }else{
      if(out_nodes.back()!=nd1){
        int l = indptr.size();
        indptr.push_back(indptr[l-1]);
        out_nodes.push_back(nd1);
      }
    }


  }

  inline void add_from_node(long nd1, int gpu_id){
      if((from_ids[gpu_id].size() != 0 ) && (from_ids[gpu_id].back() == nd1)){
        return;
      }
      from_ids[gpu_id].push_back(nd1);
     if(((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
        owned_out_nodes.push_back(nd1);
     }

  }

  inline void add_to_node(long nd1, int gpu_id){
    if((to_ids[gpu_id].size() != 0 ) && (to_ids[gpu_id].back() == nd1)){
      return;
    }
    to_ids[gpu_id].push_back(nd1);

    // if(((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
    //  owned_out_nodes.push_back(nd1);
    // }
  }


  inline void add_edge(int nd1, int nd2, bool islocal){
    // if(nd1 == nd2){
    //   std::cout << "Should never happen" << nd1 <<"\n";
    //   assert(false);
    // }
    if(islocal && ((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
		  owned_out_nodes.push_back(nd1);
      }
      if(indptr.size() == 0){
        indptr.push_back(0);
        indptr.push_back(0);
        if((out_nodes.size()==0) || (out_nodes.back() != nd1)){
          out_nodes.push_back(nd1);
        }
      }else{
        if(out_nodes.back()!=nd1){
          int l = indptr.size();
          indptr.push_back(indptr[l-1]);
          out_nodes.push_back(nd1);
        }
      }
      in_nodes.push_back(nd2);
      expand_indptr.push_back(nd1);
      indices.push_back(nd2);
      int l = indptr.size();
      indptr[l-1] = indptr[l-1] + 1;
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
    expand_indptr.clear();
    in_nodes.clear();
    out_nodes.clear();
    owned_out_nodes.clear();
    cached_node_ids.clear();
    missing_node_ids.clear();
    in_degree.clear();
    num_in_nodes = 0;
    num_out_nodes = 0;
  }

  void reorder(DuplicateRemover* dr);

  void reorder_lastlayer(DuplicateRemover *dr, vector<int>& gpu_order, int gpu_capacity);

  void debug_vector(string str, std::vector<long>& data, std::ostream& stream){
    stream << str <<":";
    int c = 0;
    for(long d: data){
      stream << d << " ";
      c ++ ;
      if (c % 20 == 0 ) stream <<  "\n";
    }
    stream <<"\n";
  }
  void debug(){
    std::ostream &out = std::cout ;
    std::cout << "BiPartitie" << "\n";
    out << "num_ in nodes" << num_in_nodes <<"\n";
    out << "nun_ out nodes" << num_out_nodes <<"\n";
    debug_vector("in_nodes", in_nodes, out);
    debug_vector("out_nodes", out_nodes, out);
    debug_vector("indptr", indptr, out);
    debug_vector("expand_indptr", expand_indptr, out);
    debug_vector("indices", indices, out);
    debug_vector("in_degree", in_degree, out);
    debug_vector("missing_node_ids", missing_node_ids, out);
    debug_vector("owned_out_nodes", owned_out_nodes, out);

    // vector<long> from_ids[4];
    // vector<long> to_ids[4];
    //
    // // Used for self attention.
    // vector<long> self_ids_in;
    // vector<long> self_ids_out;

    // int gpu_id = -1;
    // debug_vector("in_nodes", in_nodes, std::cout);
    // std::cout << "gpu" << gpu_id << "in_nodes" << in_nodes.size() << "out_nodes" << out_nodes.size() \
		//   << "owned_out_nodes" \
		//   << owned_out_nodes.size() << "\n";
  }
};
