#include "samplers/samplers.h"
#include <iostream>

void NeighbourSampler::layer_sample(std::vector<long> &in,
   std::vector<long> &in_degrees, std::vector<long> &offsets,
    std::vector<long> &indices){
      offsets.clear();
      indices.clear();
      in_degrees.clear();
      offsets.push_back(indices.size());
      for(long nd1: in){
          // Add self loop
          // In GCN self value is needed after normalization.
          // In GAT self value needed for handling zero degree vertices.
          indices.push_back(nd1);
          long nbs = dataset->indptr[nd1+1] - dataset->indptr[nd1];
          int offset = dataset->indptr[nd1];
          int in_degree = 0;
          if((nbs < fanout) || (this->deterministic)){
            in_degree = nbs;
            for(int i=0;i<nbs;i++){
              indices.push_back(dataset->indices[offset + i]);
            }
          }else{
            in_degree = fanout;
            for(int i=0;i<fanout;i++){
              int rand_nb = this->random_number_engine()%nbs;
              indices.push_back(dataset->indices[offset + rand_nb ]);
            }
          }
          if (in_degree == 0){
            // To avoid zero/zero division when divide
            in_degree = 1;
          }
          in_degrees.push_back(in_degree);
          offsets.push_back(indices.size());
      }
  }

void NeighbourSampler::sample(std::vector<long> &target_nodes, Sample &s){
  s.block[0]->clear();
  s.block[0]->layer_nds = target_nodes;
  for(int i=1;i<s.num_layers+1;i++){
    s.block[i]->clear();
    layer_sample(s.block[i-1]->layer_nds,s.block[i]->in_degree,
            s.block[i]->offsets, s.block[i]->indices);
    s.block[i]->layer_nds = s.block[i]->indices;

    dr->order_and_remove_duplicates(s.block[i]->layer_nds);
    dr->replace(s.block[i]->indices);
    dr->clear();
  }
}
