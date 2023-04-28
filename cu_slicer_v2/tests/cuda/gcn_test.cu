#include "../../graph/sliced_sample.h"
#include "../../graph/sample.h"
#include <iostream>
#include "../../graph/bipartite.h"
#include <vector>
#include <assert.h>
#include "test.h"
#include <bits/stdc++.h>
#include<tuple>
// Test gcn and gat
using namespace std;

void aggregate_gcn(std::vector<long> layer_out_nds, \
	       		std::vector<long>  layer_in_nds, \
        			std::vector<long> offsets,
			       std::vector<long> indices, \
         			std::vector<long> degree,  \
        			std::vector<int> &in, \
			       	std::vector<int> &out){
  out.clear();
  for(int i=0;i < (int)offsets.size()-1; i++){
    int start = offsets[i];
    int end = offsets[i+1];
    int nbs = 0;
    int self = 0;
    long dest_nd = layer_out_nds[i];

    for(int j = start; j < end; j++){
      int src_nd = layer_in_nds[indices[j]];
      if ((src_nd == dest_nd)){
	       std::cout << "Found self node which is never the case\n";
        self = in[indices[j]];
      }else{
        // std::cout << "(" << indices[j] <<":"<< in[indices[j]]<<")";
	      nbs += in[indices[j]];
      }
    }
    out.push_back((nbs/degree[i]) + in[i]);
    // std::cout << "(" << i <<":" <<out[i] <<")";
    //out.push_back((nbs/degree[i]) + self);

  }
}


// Sample without any reordering.
// returns sum of flow up after all layers.
std::vector<int> naive_flow_up_sample_gcn(Sample &s, int number_of_nodes){
   std::vector<int> in_f;
   std::vector<int> out_f;
   // num lyaers = 3
   // since tehre is null layer iinitially ()
   for(auto nd : s.block[s.num_layers]->layer_nds.to_std_vector()){
     // in_f.push_back(nd);
     in_f.push_back(nd % 10);
   }

   for(int i=s.num_layers - 1; i >=0; i--){
     aggregate_gcn(s.block[i]->layer_nds.to_std_vector(), s.block[i+1]->layer_nds.to_std_vector(), \
          s.block[i+1]->offsets.to_std_vector(), s.block[i+1]->indices.to_std_vector(), \
          s.block[i+1]->in_degree.to_std_vector(), in_f, out_f);
	   in_f.swap(out_f);
   }
   int sd = 0;
   for(int nd: in_f){
	   sd += ((nd) );
   }
   // return sd;
   return in_f;
}

void aggregate(std::vector<int> &out, std::vector<int> &in,
        std::vector<long> indptr, std::vector<long> indices){
    if(indptr.size()<2)return;
    if(indptr.size()>1){
        assert(out.size() == indptr.size()-1);
    }

    for(int i=0;i< (int) indptr.size()-1;i ++){
      int off_start = indptr[i];
      int off_end = indptr[i+1];
      int t = 0;
      for(int off = off_start; off < off_end; off ++ ){
          t += in[indices[off]];

          if(in[indices[off]] < 0){
            std::cout <<"Incorrect read "<<  indices[off] << " " << in[indices[off]] <<"\n";
          }
          // std::cout << "(" << in[indices[off]]<<":"<<indices[off]<<")";
          assert(in[indices[off]] >= 0);
      }

      out[i] = t;
    }
}
// A Bit Confusing here.

void shuffle(std::vector<long> from_ids,  std::vector<int> &to,
         std::vector<int> from,  int start, int end){
  assert(from_ids.size() == (end - start));
  for(int i=0; i< (int) from_ids.size(); i++){
    to[from_ids[i]] += from[start + i];
    }
}

void pull_own_node(BiPartite *bp,
      std::vector<int> &out, std::vector<int> &in){
        if(bp->self_ids_offset != bp->out_degree_local.size()){
          std::cout << "Check se" << bp->self_ids_offset << " " << bp->out_degree_local.size() <<"\n";
        }
  assert(bp->self_ids_offset == bp->out_degree_local.size());
  for(int i=0; i < bp->self_ids_offset; i++){
    if (i == 41){
      std::cout <<"local compute" << out[i] << ":" << bp->out_degree_local[i] <<":" << in[i] <<"\n";
    }
      out[i] = (out[i] /bp->out_degree_local[i]) + in[i];
    }
}


// Partitioned flow must have same output.
  std::vector<std::tuple<int, int>>  sample_flow_up_ps(PartitionedSample &s,
    std::vector<NDTYPE> storage_map[8], int num_gpus){
  // refresh storage map with local_ids.
  std::vector<int> in[8];
  std::vector<int> out[8];
  std::vector<int> remote_out[8];
  std::vector<long> cache_hit_from[8];
  std::vector<long> cache_hit_to[8];
  std::vector<long> cache_miss_from[8];
  std::vector<long> cache_miss_to[8];
  for(int i=0;i<num_gpus; i++ ){
     in[i].clear();
     int cache_hit = s.cache_hit_to[i].size();
     int cache_miss = s.cache_miss_to[i].size();
     in[i].resize(cache_hit + cache_miss);
     for(int j=0; j < cache_hit; j++) {
         // std::cout << "(" << s.cache_hit_to[i][j] << ":" << storage_map[i][s.cache_hit_from[i][j]] <<")";
          // in[i][s.cache_hit_to[i][j]] = storage_map[i][s.cache_hit_from[i][j]];
          in[i][s.cache_hit_to[i][j]] = storage_map[i][s.cache_hit_from[i][j]]%10;
        }
     for(int j=0; j < cache_miss; j++) {
           // in[i][s.cache_miss_to[i][j]] = s.cache_miss_from[i][j];
           in[i][s.cache_miss_to[i][j]] = s.cache_miss_from[i][j]%10;

     }
  }
  for(int i =  s.num_layers-1  ; i>=0; i--){
    // Bipartite local aggregation.

    PartitionedLayer &layer = s.layers[i];
    // PULL
    for(int j=0;j < num_gpus;j ++){
      BiPartite *bp = layer.bipartite[j];
      std:cout << "error" << in[j].size() << " " << bp->in_nodes_local.size() <<" " << bp->num_in_nodes_pulled <<"\n";
      assert(in[j].size() == bp->in_nodes_local.size());
      int new_size = bp->num_in_nodes_local + bp->num_in_nodes_pulled;

      in[j].resize(new_size);
      for(int pull_from = 0; pull_from <num_gpus; pull_from ++){
        if(pull_from == j)continue;
        int start = bp->pull_from_offsets[pull_from];
        int end = bp->pull_from_offsets[pull_from + 1];
        if(end - start == 0)continue;
        std::vector<long> push_to = layer.bipartite[pull_from]->pull_to_ids[j].to_std_vector();
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
    for(int j=0; j<num_gpus; j++ ){
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
        aggregate(out[j], in[j], bp->indptr_L.to_std_vector(), bp->indices_L.to_std_vector());
        aggregate(remote_out[j], in[j], bp->indptr_R.to_std_vector(), bp->indices_R.to_std_vector());
    }
    // PUSH
    for(int from = 0; from < num_gpus; from ++) {
      for(int to = 0; to < num_gpus ; to++) {
          if(from != to){
            int start = layer.bipartite[from]->to_offsets[to];
            int end = layer.bipartite[from]->to_offsets[to + 1];
            shuffle(layer.bipartite[to]->push_from_ids[from].to_std_vector(), out[to],\
              remote_out[from], \
                    start , end);
          }
      }
    }
    // Pull locally and add degree
    for(int j = 0; j < num_gpus; j++){
      pull_own_node(layer.bipartite[j], out[j], in[j]);
    }
    // Swap
    for(int j=0;j<num_gpus; j++){
      out[j].resize(layer.bipartite[j]->num_out_local);
      out[j].swap(in[j]);
    }
  }
  int sss = 0;
  // Return final sum
  std::vector<std::tuple<int, int>> t;
  for(int i=0;i < num_gpus;i++){
    int ss = 0;
    PartitionedLayer &layer = s.layers[0];
    auto v = layer.bipartite[i]->out_nodes_local.to_std_vector();
    for(int j = 0 ; j < in[i].size() ; j++){
      t.push_back(make_tuple(v[j],in[i][j]));
      ss += in[i][j];
    }
    sss += ss;
  }

  sort(t.begin(), t.end(), [&](auto a, auto b){
    return get<0>(a) < get<0>(b);
  });


  // return sss;
  return t;
}

void test_sample_partition_consistency(Sample &s, PartitionedSample &ps,
  std::vector<NDTYPE> local_storage[8], int gpu_capacity[8],
    int num_nodes, int num_gpus){
    auto correct = naive_flow_up_sample_gcn(s, num_nodes);
    // std::cout << "Correct answer is " << correct << "\n";

    for(int i=0;i<num_gpus; i++){
      assert(local_storage[i].size() == gpu_capacity[i]);
    }
    auto out = sample_flow_up_ps(ps, local_storage, num_gpus);
    for(int i = 0; i <correct.size(); i++){
      if(correct[i] != get<1>(out[i])){
        std::cout << "Miss" << get<0>(out[i]) <<" expect "<< \
            get<1>(out[i]) << "original"<< correct[i] <<"\n";
            assert(false);
      }
    }
    // std::cout << correct << "==" << out << "\n";
    // assert(correct == out);

}
