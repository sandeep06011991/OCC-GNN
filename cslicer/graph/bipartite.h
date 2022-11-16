#include "util/duplicate.h"
#include <vector>
#include <iostream>
using namespace std;
#pragma once

// ================BiPartite Dest Nodes Order========================
// |             |             |             |             |
// |   Local     | remote[1]   | remote[2]   | remote[3]   |
// |             |             |             |             |
// ==================================================================
// ================BiPartite Src Nodes Order=========================
// |             |             |             |             |
// |   Local     | Pulled[1]   | Pulled[2]   | pulled[3]   |
// |             |             |             |             |
// ==================================================================
class BiPartite{

public:
  // sorted and no duplicate nodes using graph_ids.
  // Used to find mapping between local id and global id
  // nd2 is in_nodes as sampling is top down, but flow is bottom up.
  // Follows global order.
  // Built during REORDERING
  vector<long> in_nodes;
  vector<long> pulled_in_nodes;
  vector<long> out_nodes_remote;
  int remote_sizes[4];
  int pull_offsets[4];
  // Built during slicing.
  vector<long> out_nodes_local;
  vector<long> out_degree_local;

  // Built during REORDERING
  // Num in nodes can be much larger then in the actual graph, if we use
  // cached node order on the gpu.
  int num_in_nodes;
  // out_nodes_local + out_nodes_remote
  int num_out_nodes;

  // Built during REORDERING
  // built locally
  vector<long> indptr_L;
  vector<long> indices_L;
  vector<long> indptr_R;
  vector<long> indices_R;

  // Build during slicing
  vector<long> indptr[4];
  vector<long> indices[4];
  vector<long> to_ids[4];

  // Built during reordering
  vector<long> from_ids[4];
  int to_offsets[5];

  // Built during slicing
  vector<long> part_in_nodes[4];

  // Built during reoridering
  vector<long> push_to_ids[4];
  int pull_from_sizes[4];

  // Used for self attention.
  // Built during re-ordering.
  // Self Nodes are never assigned to a different partition and never pulled therefore.
  int self_ids_offset;

  int gpu_id = -1;

  BiPartite(int gpu_id){
    this->gpu_id = gpu_id;
  }

  void add_local_out_node(long nd_dest, int degree){
    out_nodes_local.push_back(nd_dest);
    out_degree_local.push_back(degree);
    in_nodes.push_back(nd_dest);
  }

  // Single function for all remote graphs.
  void merge_graph(vector<long> &edges, long nd_dest, int partition_id){
      // Both GCN and GAT need in node
      vector<long> indptr & = indptr[partition_id];
      vector<long> indices & = indices[partition_id];
      vector<long> to_ids & = to_ids[partition_id];

      if(indptr.sizes() == 0)indptr.push_back(0);
      indptr.push_back(indptr[indptr.size()-1]+edges.size());
      indices.insert(indices.end(), edges.begin(), edges.end());
      to_ids.insert(nd_dest);
  }


  void merge_in_nodes(vector<long> pull_nodes, long p_id){
      pull_from_ids[p_id].insert(pull_from_ids.end(), pull_nodes.begin(), pull_nodes.end());
  }

  // inline void add_self_edge(long nd1, int degree){
  //   if((self_ids_in.size()!=0) && (self_ids_in.back() == nd1)){
  //     return;
  //   }
  //   self_ids_in.push_back(nd1);
  //   self_ids_out.push_back(nd1);
  //   in_degree.push_back(degree);
  //   assert(degree > 0);
  //   // in_nodes and out nodes are pushed here instead of at add_edge
  //   // as self loops are not counted in the main graph
  //   //
  //   in_nodes.push_back(nd1);
  //   owned_out_nodes.push_back(nd1);
  //   if(indptr.size() == 0){
  //     indptr.push_back(0);
  //     indptr.push_back(0);
  //     if((out_nodes.size()==0) || (out_nodes.back() != nd1)){
  //       out_nodes.push_back(nd1);
  //     }
  //   }else{
  //     if(out_nodes.back()!=nd1){
  //       int l = indptr.size();
  //       indptr.push_back(indptr[l-1]);
  //       out_nodes.push_back(nd1);
  //     }
  //   }
  //
  //
  // }
  //
  // inline void add_from_node(long nd1, int gpu_id){
  //     if((from_ids[gpu_id].size() != 0 ) && (from_ids[gpu_id].back() == nd1)){
  //       return;
  //     }
  //     from_ids[gpu_id].push_back(nd1);
  //    if(((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
  //       owned_out_nodes.push_back(nd1);
  //    }
  //
  // }
  //
  // inline void add_to_node(long nd1, int gpu_id){
  //   if((to_ids[gpu_id].size() != 0 ) && (to_ids[gpu_id].back() == nd1)){
  //     return;
  //   }
  //   to_ids[gpu_id].push_back(nd1);
  //
  //   // if(((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
  //   //  owned_out_nodes.push_back(nd1);
  //   // }
  // }
  //
  //
  // inline void add_edge(int nd1, int nd2, bool islocal){
  //   // if(nd1 == nd2){
  //   //   std::cout << "Should never happen" << nd1 <<"\n";
  //   //   assert(false);
  //   // }
  //   if(islocal && ((owned_out_nodes.size() == 0) || (owned_out_nodes.back() != nd1))){
	// 	  owned_out_nodes.push_back(nd1);
  //     }
  //     if(indptr.size() == 0){
  //       indptr.push_back(0);
  //       indptr.push_back(0);
  //       if((out_nodes.size()==0) || (out_nodes.back() != nd1)){
  //         out_nodes.push_back(nd1);
  //       }
  //     }else{
  //       if(out_nodes.back()!=nd1){
  //         int l = indptr.size();
  //         indptr.push_back(indptr[l-1]);
  //         out_nodes.push_back(nd1);
  //       }
  //     }
  //     in_nodes.push_back(nd2);
  //     expand_indptr.push_back(nd1);
  //     indices.push_back(nd2);
  //     int l = indptr.size();
  //     indptr[l-1] = indptr[l-1] + 1;
  // }

  void refresh(){
    assert(False);
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
