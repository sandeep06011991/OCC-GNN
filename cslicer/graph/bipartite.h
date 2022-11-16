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
  // Built during slicing.
  vector<long> out_nodes_local;
  vector<long> out_degree_local;

  // Built during REORDERING
  // Num in nodes can be much larger then in the actual graph, if we use
  // cached node order on the gpu.
  int num_in_nodes;
  // out_nodes_local + out_nodes_remote
  // Used in graph construction
  int num_out_local;
  int num_out_remote;


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
  // vector<long> part_in_nodes[4];

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
      vector<long> & indptr_c  = indptr[partition_id];
      vector<long> & indices_c  = indices[partition_id];
      vector<long> & to_ids_c  = to_ids[partition_id];

      if(indptr_c.size() == 0)indptr_c.push_back(0);
      indptr_c.push_back(indptr_c[indptr_c.size()-1]+edges.size());
      indices_c.insert(indices_c.end(), edges.begin(), edges.end());
      to_ids_c.insert(nd_dest);
  }


  void merge_in_nodes(vector<long> pull_nodes, long p_id){
      pull_from_ids[p_id].insert(pull_from_ids.end(), pull_nodes.begin(), pull_nodes.end());
  }


  void refresh(){
    in_nodes.clear();
    pulled_in_nodes.clear();
    out_nodes_remote.clear();
    out_nodes_local.clear();
    out_degree_local.clear();
    num_in_nodes = 0;
    num_out_local = 0;
    num_out_remote = 0;
    for(int i=0;i<4;i++){
      from_ids[i].clear();
      to_ids[i].clear();
      remote_sizes[i] = 0;
      pull_sizes[i] = 0;
      indptr[i].clear();
      indices[i].clear();
      to_ids[i].clear();
      from_ids[i].clear();
      to_offsets[i] = 0;
      push_to_ids[i].clear();
      pull_from_sizes[i] = 0;
    }
    indptr_L.clear();
    indices_L.clear();
    indptr_R.clear();
    indices_R.clear();
    self_ids_offset = 0;
  }

  void reorder_local(DuplicateRemover *dr);

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
    out << "nun_ out nodes" << num_out_local <<"\n";
    debug_vector("in_nodes", in_nodes, out);
    debug_vector("pulled in_nodes", pulled_in_nodes, out);
    debug_vector("out nodes remote", out_nodes_remote, out);

  }
};
