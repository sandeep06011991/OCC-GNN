#include "util/duplicate.h"
#include <vector>
#include <iostream>
#include <thrust/device_vector.h>
#include <util/cuda_utils.h>
using namespace std;
#pragma once

// ================BiPartite Dest Nodes Order========================
// |             |             |             |             |
// |   Local_Out | remote[1]   | remote[2]   | remote[3]   |
// |             |             |             |             |
// ==================================================================
// ================BiPartite Src Nodes Order=========================
// |             |             |             |             |
// |   Local_Out | Local In    |  Pulled[1]  | Pulled[2]   | pulled[3]
// |             |             |             |             |
// ==================================================================
class BiPartite{

public:
  // sorted and no duplicate nodes using graph_ids.
  // Used to find mapping between local id and global id
  // nd2 is in_nodes as sampling is top down, but flow is bottom up.
  // Follows global order.
  // Built during REORDERING
  // All locally present in nodes
  thrust::device_vector<long> in_nodes;
  // Self nodes that are reordered first
  thrust::device_vector<long> self_in_nodes;
  thrust::device_vector<long> pulled_in_nodes;
  thrust::device_vector<long> out_nodes_remote;
  // out nodes_remote[gpu_i] = out_nodes_remote[remote_offsets[i]:[i+1]]
  // Stick to this throughout.

  // Built during slicing.
  thrust::device_vector<long> out_nodes_local;
  thrust::device_vector<long> out_degree_local;

  // Built during REORDERING
  // Num in nodes can be much larger then in the actual graph, if we use
  // cached node order on the gpu.
  int num_in_nodes_local = 0;
  int num_in_nodes_pulled = 0;
  // out_nodes_local + out_nodes_remote
  // Used in graph construction
  int num_out_local = 0;
  int num_out_remote = 0;


  // Built during REORDERING
  // built locally
  thrust::device_vector<long> indptr_L;
  thrust::device_vector<long> indices_L;
  thrust::device_vector<long> indptr_R;
  thrust::device_vector<long> indices_R;

  // Build during slicing
  thrust::device_vector<long> indptr_[MAX_DEVICES];
  thrust::device_vector<long> indices_[MAX_DEVICES];


  // Built during reordering
  // Destination vertices out nodes, that have to be send will be populated locally at to_ids_
  // After reordering destination partition stores to_ids as offsets and from_ids are populated at src.
  thrust::device_vector<long> push_from_ids[MAX_DEVICES];
  int to_offsets[MAX_DEVICES + 1];
  thrust::device_vector<long> push_to_ids_[MAX_DEVICES];
  thrust::device_vector<long> in_nodes_[MAX_DEVICES];
  // Built during slicing
  // vector<long> part_in_nodes[MAX_DEVICES];

  // In nodes that have be pulled from neighbour partition
  // These ids will be reordered so that the src partition knows what values to send
  // which are stored in push_to_ids, the dest partition only stores the offsets where
  // to place these recieved values.
  thrust::device_vector<long> pull_to_ids[MAX_DEVICES];
  int pull_from_offsets[MAX_DEVICES + 1];
  thrust::device_vector<long> pull_from_ids_[MAX_DEVICES];

  // Used for self attention.
  // Built during re-ordering.

  // Self Nodes are never assigned to a different partition and never pulled therefore.
  int self_ids_offset = 0;

  int gpu_id = -1;
  int num_gpus = num_gpus;
  BiPartite(int gpu_id, int num_gpus){
    this->gpu_id = gpu_id;
    this->num_gpus = num_gpus;
    refresh();
  }



  // Single function for all remote graphs.
  // Deprecated not used in cuda version
  // void merge_graph(vector<long> &edges, long nd_dest, int partition_id){
  //     // Both GCN and GAT need in node
  //     thrust::device_vector<long> & indptr_c  = indptr_[partition_id];
  //     thrust::device_vector<long> & indices_c  = indices_[partition_id];
  //     thrust::device_vector<long> & to_ids_c  = to_ids_[partition_id];
  //     if(indptr_c.size() == 0)indptr_c.push_back(0);
  //     indices_c.insert(indices_c.end(), edges.begin(), edges.end());
  //     indptr_c.push_back(indices_c.size());
  //     to_ids_c.push_back(nd_dest);
  // }

  //
  // void merge_pull_nodes(vector<long> pull_nodes, long p_id){
  //     pull_from_ids_[p_id].insert(pull_from_ids_[p_id].end(), pull_nodes.begin(), pull_nodes.end());
  // }
  //
  // void merge_local_in_nodes(vector<long> in_nodes){
  //     this->in_nodes.insert(this->in_nodes.end(), in_nodes.begin(), in_nodes.end());
  // }


  void refresh(){
    in_nodes.clear();
    pulled_in_nodes.clear();
    out_nodes_remote.clear();
    out_nodes_local.clear();
    out_degree_local.clear();
    num_in_nodes_local = 0;
    num_in_nodes_pulled = 0;
    num_out_local = 0;
    num_out_remote = 0;
    to_offsets[0] = 0;
    pull_from_offsets[0] = 0;
    for(int i=0;i<this->num_gpus;i++){
      push_from_ids[i].clear();
      push_to_ids_[i].clear();
      indptr_[i].clear();
      indices_[i].clear();
      to_offsets[i + 1] = 0;
      pull_from_offsets[i + 1] = 0;
      pull_to_ids[i].clear();
      pull_from_ids_[i].clear();
    }
    indptr_L.clear();
    indices_L.clear();
    indptr_R.clear();
    indices_R.clear();
    self_ids_offset = 0;
    self_in_nodes.clear();
  }

  void reorder_local(DuplicateRemover *dr);

  void debug_vector(string str, thrust::device_vector<long>& data, std::ostream& stream){
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
    std::cout << "BiPartitie############" << gpu_id <<  "\n";
    debug_vector("in_nodes", in_nodes, out);
    debug_vector("self in  nodes", self_in_nodes, out);
    debug_vector("pulled in_nodes", pulled_in_nodes, out);
    debug_vector("out nodes remote", out_nodes_remote, out);
    debug_vector("out nodes local", out_nodes_local, out);
    debug_vector("out degree local", out_degree_local, out);
    std::cout << "To";
    for(int i=0;i<this->num_gpus;i++){
      std::cout << to_offsets[i+1] << " ";
    }
    std::cout << "\n From";
    for(int i=0;i<this->num_gpus;i++){
      std::cout << pull_from_offsets[i+1] << " ";
    }
    for(int i=0;i<this->num_gpus;i++){
        std::cout <<  i << ":\n";
        debug_vector("push_from_ids", push_from_ids[i], out);
        debug_vector("push_to_ids_", push_to_ids_[i], out);
        debug_vector("indptr_",indptr_[i], out);
        debug_vector("indices_", indices_[i], out);
        debug_vector("pull_to_ids", pull_to_ids[i], out);
        debug_vector("pull_from_ids", pull_from_ids_[i], out);
      }
    debug_vector("indptr_L", indptr_L, out);
    debug_vector("indices_L", indices_L, out);
    debug_vector("indptr_R", indptr_R, out);
    debug_vector("indices_R", indices_R, out);
    std::cout << "End \n";
  }
};
