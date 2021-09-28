#pragma once
#include<algorithm>
// creates minibatches of given size by shuffling target node
// with full 2 hop neighbourhood.
class TwoHopNoSample{

  // Full graph
  int no_nodes;
  int no_edges;
  int * ind_ptr;
  int * indices;
  int * target_nodes;
  // Instantitate from sample 1-n

  // bipartite graph 1
  int * ind_ptr_l1;
  int * indices_l2;

  // Features of the last sampled_hop
  float * sampled_features;
  int next_minibatch;
public:

  TwoHopNoSample(int no_nodes, int no_edges, int *ind_ptr, int *indices, int max_batch_size){
    this->no_nodes = no_nodes;
    this->no_edges = no_edges;
    this->ind_ptr = ind_ptr;
    this->indices = indices;
    this->target_nodes = (int *)malloc(sizeof(int) * no_nodes);
    for(int i =0;i<no_nodes;i++){
      this->target_nodes[i] = i;
    }
    this->next_minibatch=0;

  }

  void shuffle(){
    std::random_shuffle(&this->target_nodes[0], &this->target_nodes[this->no_nodes]);
    next_minibatch = 0;
  }

  void sample();

};
