#pragma once
#include<algorithm>

// creates minibatches of given size by shuffling target node
// with full 2 hop neighbourhood.

class Graph{
  int num_nodes;
  int num_edges;
  int *indptr;
  int *indices;
};

struct ComputeGraph{
  int num_nodes_out;
  int num_nodes_in;
  vector<int> *indptr;
  vector<int> *indices;
};


class TwoHopNoSample{

  Graph graph;
  // Full graph
  int no_nodes;
  int no_edges;
  // Instantitate from sample 1-n

  // bipartite graph 1
  ComputeGraph l1;
  ComputeGraph l2;
  // Features of the last sampled_hop
  float * sampled_features;

  int next_minibatch;
  int minibatch_size;
  int * target_nodes;

public:

  TwoHopNoSample(int no_nodes, int no_edges, int *ind_ptr, int *indices, int max_batch_size){
    this->graph.num_nodes = num_nodes;
    this->graph.num_edges = num_edges;
    this->graph.ind_ptr = ind_ptr;
    this->indices = indices;
    this->no_nodes = no_nodes;
    this->no_edges = no_edges;
    this->target_nodes = (int *)malloc(sizeof(int) * no_nodes);

    for(int i =0;i<no_nodes;i++){
      this->target_nodes[i] = i;
    }
    this->next_minibatch=0;
    this->minibatch_size = max_batch_size;
    shuffle();
  }

  void shuffle(){
    std::random_shuffle(&this->target_nodes[0], &this->target_nodes[this->no_nodes]);
    next_minibatch = 0;
  }

  void sample(){
    int * tgt = this->target_nodes[this->next_minibatch];
    int no_nodes =  minibatch_size;
    if(this->next_minibatch + no_nodes > this->no_nodes){
      no_nodes = this->no_nodes - this->next_minibatch;
    }
  }

  void sample_from_target(int * target_vertices, int no_targets);

  void sample_from_hop(ComputeGraph &l,int *target_vertices, int no_targets);

};
