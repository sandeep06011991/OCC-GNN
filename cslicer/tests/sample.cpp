#include "graph/sliced_sample.h"
#include "graph/sample.h"
#include <iostream>
#include "graph/bipartite.h"
#include <vector>
#include <assert.h>
using namespace std;

void aggregate(std::vector<long>& layer_nds, \
        std::vector<long> &offsets, std::vector<long> &indices, \
         std::vector<long> &degree, \
        std::vector<int> &in, std::vector<int> &out, bool first_layer){
  for(int i=0;i < offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int src = layer_nds[i];
    int t = 0;
    int acc = 0;
    for(int j = start; j < end; j++){
      int dest = indices[j];
      acc = dest;
      if(!first_layer){
        acc = in[dest];
      }
      if (src == dest){
        acc = acc * degree[i];
      }
      t  += acc;
    }
    out[src] = acc;
  }
}


// Sample without any reordering.
// returns sum of flow up after all layers.
int sample_flow_up(Sample &s, int number_of_nodes){
   std::vector<int> in_f(number_of_nodes);
   std::vector<int> out_f(number_of_nodes);
   for(int i=s.num_layers; i >=0; i--){
     bool first_layer = (i == s.num_layers);
     aggregate(s.block[i]->layer_nds, s.block[i+1]->offsets, s.block[i+1]->indices, \
          s.block[i+1]->in_degree, in_f, out_f, first_layer);
     in_f.swap(out_f);
   }
   int sd = 0;
   for(int nd: s.block[0]->layer_nds){
     sd += in_f[nd];
   }
   return sd;
}

void aggregate(vector<int> &out, vector<int> &in, BiPartite *bp){
    vector<long> &indptr = bp->indptr;
    vector<long> &indices = bp->indices;
    for(int i=0;i<indptr.size();i ++){
      int off_start = indptr[i];
      int off_end = indptr[i+1];
      int t = 0;
      for(int off = off_start; off < off_end; off ++ ){
          t += in[indices[off]];
      }
      out[i] = t;
    }
}
// A Bit Confusing here.

void shuffle(vector<long>& from_ids,  vector<int> &from_v,
         vector<long> &to_ids,  vector<int> &to_v){
  assert(from_ids.size() == to_ids.size());
  for(int i=0; i< from_ids.size(); i++){
    to_v[to_ids[i]] += from_v[from_ids[i]];
    }
}

void pull_own_node(BiPartite *bp,
      vector<int> &out, vector<int> &in){

  assert(bp->self_ids_in.size() == bp->self_ids_out.size());
  for(int i=0; i < bp->self_ids_in.size(); i++){
    out[bp->self_ids_out[i]] += bp->in_degree[i] * in[bp->self_ids_in[i]];
  }
  in.resize(bp->owned_out_nodes.size());
  for(int i=0;i< bp->owned_out_nodes.size(); i++){
    in[i] = out[bp->owned_out_nodes[i]];
  }
}


// Partitioned flow must have same output.
int sample_flow_up(PartitionedSample &s,
    std::vector<int> test_storage_map[4]){
  // refresh storage map with local_ids.
  std::vector<int> in[4];
  std::vector<int> out[4];
  for(int i=0;i<4; i++ ){
    in[i].swap(test_storage_map[i]);
  }
  for(int i =  s.num_layers-1  ; i>=0; i--){
    // Bipartite local aggregation.
    PartitionedLayer &layer = s.layers[i];
    for(int j=0; j<4; j++ ){
        out[j].resize(layer.bipartite[j]->num_out_nodes);
        aggregate(out[j], in[j], layer.bipartite[j]);
    }
    // Shuffle aggregate
    for(int from = 0; from < 4; from ++) {
      for(int to = 0; to < 4 ; to++) {
          if(from != to){
            shuffle(layer.bipartite[from]->to_ids[to],
                    out[from], layer.bipartite[to]->from_ids[from], out[to]);
          }
      }
    }
    // Pull locally and add degree
    // Slice owned node.
    for(int j = 0; j < 4; j++){
      pull_own_node(layer.bipartite[j], out[j], in[j]);
    }
  }
  int ss = 0;
  for(int i=0;i < 4;i++){
    for(int k:in[i]){
      ss += k;
    }
  }
  return ss;
}



void test_sample_partition_consistency(Sample &s, PartitionedSample &ps,
  std::vector<int> storage_map[4], int gpu_capacity[4]){
    int num_nodes = storage_map[0].size();
    int correct = sample_flow_up(s, num_nodes);
    std::cout << "correct " << correct <<"\n";
    for(int i=0;i<4;i++){
      std::vector<long> add  = ps.layers[2].bipartite[i]->missing_node_ids;
      int c = gpu_capacity[i];
      for(auto nd: add){
        storage_map[nd][i] = c;
        c ++;
      }
    }
    int out = sample_flow_up(ps, storage_map);
    assert(correct == out);
}
