#include "graph/sliced_sample.h"
#include "graph/sample.h"
#include <iostream>
#include "graph/bipartite.h"
#include <vector>
#include <assert.h>
// Test gcn and gat
using namespace std;

void aggregate_gcn(std::vector<long>& layer_out_nds, std::vector<long> & layer_in_nds, \
        std::vector<long> &offsets, std::vector<long> &indices, \
         std::vector<long> &degree,  \
        std::vector<int> &in, std::vector<int> &out){
  out.clear();
  for(int i=0;i < (int)offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int nbs = 0;
    int self = 0;
    long dest_nd = layer_out_nds[i];
    for(int j = start; j < end; j++){
      int src_nd = layer_in_nds[indices[j]];

      if (src_nd == dest_nd){
        self = in[indices[j]];
      }else{
        nbs += in[indices[j]];
      }
    }
    out.push_back((nbs/degree[i]) + self);
  }
}


// Sample without any reordering.
// returns sum of flow up after all layers.
int naive_flow_up_sample(Sample &s, int number_of_nodes){
   std::vector<int> in_f;
   std::vector<int> out_f;
   // num lyaers = 3
   // since tehre is null layer iinitially ()
   for(auto nd:s.block[s.num_layers]->layer_nds){
     // in_f.push_back(nd);
     in_f.push_back(nd);
   }
   for(int i=s.num_layers - 1; i >=0; i--){
     aggregate_gcn(s.block[i]->layer_nds, s.block[i+1]->layer_nds, \
          s.block[i+1]->offsets, s.block[i+1]->indices, \
          s.block[i+1]->in_degree, in_f, out_f);
     in_f.swap(out_f);
   }
   int sd = 0;
   for(int nd: in_f){
     sd += nd;
   }
   return sd;
}

void aggregate(vector<int> &out, vector<int> &in,
        vector<long> &indptr, vector<long> &indices){
    if(indptr.size()>0)assert(out.size() == indptr.size()-1);
    for(int i=0;i< (int) indptr.size()-1;i ++){
      int off_start = indptr[i];
      int off_end = indptr[i+1];
      int t = 0;
      for(int off = off_start; off < off_end; off ++ ){
          t += in[indices[off]];
          // std::cout << "adding" << in[indices[off]] <<"\n";
          if(in[indices[off]] < 0){
            std::cout <<"Incorrect read "<<  indices[off] << " " << in[indices[off]] <<"\n";
          }
          assert(in[indices[off]] >= 0);
      }
      out[i] = t;
    }
}
// A Bit Confusing here.

void shuffle(vector<long>& from_ids,  vector<int> &to,
         vector<int>& from,  int start, int end){
  assert(from_ids.size() == (end - start));
  for(int i=0; i< (int) from_ids.size(); i++){
    to[from_ids[i]] += from[start + i];
    }
}

void pull_own_node(BiPartite *bp,
      vector<int> &out, vector<int> &in){
  assert(bp->self_ids_offset == bp->out_degree_local.size());
  for(int i=0; i < bp->self_ids_offset; i++){
    out[i] = (out[i] /bp->out_degree_local[i])  + (in[i]);
  }
}


// Partitioned flow must have same output.
int sample_flow_up_ps(PartitionedSample &s,
    std::vector<int> storage_map[4]){
  // refresh storage map with local_ids.
  std::vector<int> in[4];
  std::vector<int> out[4];
  std::vector<int> remote_out[4];
  std::vector<long> cache_hit_from[4];
  std::vector<long> cache_hit_to[4];
  std::vector<long> cache_miss_from[4];
  std::vector<long> cache_miss_to[4];
  for(int i=0;i<4; i++ ){
     in[i].clear();
     int cache_hit = s.cache_hit_to[i].size();
     int cache_miss = s.cache_miss_to[i].size();
     in[i].resize(cache_hit + cache_miss);
     for(int j=0; j < cache_hit; j++) {
          in[i][s.cache_hit_to[i][j]] = storage_map[i][s.cache_hit_from[i][j]];
     }
     for(int j=0; j < cache_miss; j++) {
           in[i][s.cache_miss_to[i][j]] = s.cache_miss_from[i][j];
     }
  }
  for(int i =  s.num_layers-1  ; i>=0; i--){
    // Bipartite local aggregation.

    PartitionedLayer &layer = s.layers[i];
    // PULL
    for(int j=0;j < 4;j ++){
      BiPartite *bp = layer.bipartite[j];
      assert(in[j].size() == bp->in_nodes.size());
      int new_size = bp->num_in_nodes_local + bp->num_in_nodes_pulled;
      in[j].resize(new_size);
      for(int pull_from = 0; pull_from <4; pull_from ++){
        if(pull_from == j)continue;
        int start = bp->pull_from_offsets[pull_from];
        int end = bp->pull_from_offsets[pull_from + 1];
        if(end - start == 0)continue;
        vector<long> &push_to = layer.bipartite[pull_from]->push_to_ids[j];
        for(int k=0;k<push_to.size() ; k++){
            in[j][bp->num_in_nodes_local + start + k] = in[pull_from][push_to[k]];
        }
      }
      // std::cout << "Input nodes" << j <<"\n";
      // for(int i = 0;i < bp->in_nodes.size() ; i++){
      //   std::cout << bp->in_nodes[i] << ":" << in[j][i] <<"\n";
      // }
      //
      // for(int i = 0;i < bp->num_in_nodes_pulled ; i++){
      //   std::cout << bp->pulled_in_nodes[i] << ":" << in[j][bp->num_in_nodes_local + i] <<"\n";
      // }
    }
    // AGGREGATE

    // layer.debug();
    for(int j=0; j<4; j++ ){
        BiPartite *bp = layer.bipartite[j];
        if(bp->num_out_local>0){
          out[j].resize(bp->num_out_local);
        }else{
          out[j].clear();
        }
        if(bp->num_out_remote>0){
          remote_out[j].resize(bp->num_out_remote);
        }else{
          remote_out[j].clear();
        }
        aggregate(out[j], in[j], bp->indptr_L, bp->indices_L);
        aggregate(remote_out[j], in[j], bp->indptr_R, bp->indices_R);
    }
    // PUSH
    for(int from = 0; from < 4; from ++) {
      for(int to = 0; to < 4 ; to++) {
          if(from != to){
            int start = layer.bipartite[from]->to_offsets[to];
            int end = layer.bipartite[from]->to_offsets[to + 1];
            shuffle(layer.bipartite[to]->from_ids[from], out[to], remote_out[from], \
                    start , end);
          }
      }
    }
    // Pull locally and add degree
    for(int j = 0; j < 4; j++){
      pull_own_node(layer.bipartite[j], out[j], in[j]);
    }
    // Swap
    for(int j=0;j<4; j++){
      out[j].resize(layer.bipartite[j]->num_out_local);
      out[j].swap(in[j]);
    }
  }
  int sss = 0;
  // Return final sum
  for(int i=0;i < 4;i++){
    int ss = 0;
    for(int k:in[i]){
      ss += k;
    }
    sss += ss;
  }
  return sss;
}



void test_sample_partition_consistency(Sample &s, PartitionedSample &ps,
  std::vector<int> local_storage[4], int gpu_capacity[4], int num_nodes){
    int correct = naive_flow_up_sample(s, num_nodes);
    for(int i=0;i<4; i++){
      assert(local_storage[i].size() == gpu_capacity[i]);
    }
    std::cout <<"reached here !\n";
    int out = sample_flow_up_ps(ps, local_storage);
    std::cout << correct << "==" << out << "\n";
    assert(correct == out);

}
