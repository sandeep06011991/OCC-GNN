#pragma once
#include<algorithm>
#include<assert.h>
#include<samplers/sample.h>
#include<cstring>

// creates minibatches of given size by shuffling target node
// with full 2 hop neighbourhood.
// Fixme:: Move the graph class to different position.
struct Graph{
  int num_nodes;
  int num_edges;
  int *indptr;
  int *indices;
};

class TwoHopNoSampler{

  Graph graph;
  // Full graph
  int no_nodes;
  int no_edges;
  // Instantitate from sample 1-n

  // Features of the last sampled_hop
  float * full_features;
  int * full_labels;



  // Utilities which support minibatching.
  // int next_minibatch;
  int minibatch_size;
  int current_minibatch_size;
  int * target_nodes;
  int fsize;


public:
  TwoHopSample sample;
  float * batch_features = nullptr;
  int * batch_labels = nullptr;

  TwoHopNoSampler(int no_nodes, int no_edges, int *ind_ptr, int *indices,
      int max_batch_size, float *features, int * labels, int fsize){
    this->graph.num_nodes = no_nodes;
    this->graph.num_edges = no_edges;
    this->graph.indptr = ind_ptr;
    this->graph.indices = indices;
    this->no_nodes = no_nodes;
    this->no_edges = no_edges;
    this->target_nodes = (int *)malloc(sizeof(int) * no_nodes);
    this->full_features = features;
    this->full_labels = labels;
    for(int i =0;i<no_nodes;i++){
      this->target_nodes[i] = i;
    }
    // this->next_minibatch=0;
    this->minibatch_size = max_batch_size;
    this->fsize = fsize;
    // shuffle();
  }

  void shuffle(){
    std::random_shuffle(&this->target_nodes[0], &this->target_nodes[this->no_nodes]);
    // next_minibatch = 0;
  }

  void fill_batch_data(int* in_nodes, int no_in, int* out_nodes, int no_out){
    if(batch_features != nullptr){
      free(batch_features);
    }
    if(batch_labels != nullptr){
      free(batch_labels);
    }
    batch_labels = (int *)malloc(sizeof(int) * no_out);
    batch_features = (float *)malloc(sizeof(float)* no_in * this->fsize);
    for(int i=0; i < no_in;i++){
      memcpy(&batch_features[i*this->fsize],
            &full_features[in_nodes[i]*this->fsize],sizeof(float) * this->fsize);
          }
    for(int i=0; i< no_out; i++){
      batch_labels[i] = full_labels[out_nodes[i]];
    }
  }


  void get_sample(int batchId){
    assert(batchId * this->minibatch_size < this->no_nodes);
    int * tgt = &this->target_nodes[this->minibatch_size * batchId];
    int no_nodes =  minibatch_size;
    if(this->minibatch_size * (batchId + 1) > this->no_nodes){
      no_nodes = this->no_nodes - (this->minibatch_size * batchId);
    }
    this->current_minibatch_size = no_nodes;
    // Sample 2-hop neighbourhoods.
    sample.clear();

    for(int i=0;i<no_nodes;i++){
      int nd1 = tgt[i];
      int edge_start = this->graph.indptr[nd1];
      int edge_end = this->graph.indptr[nd1+1];
      if(edge_end - edge_start > 0)sample.l1.nd1.push_back(nd1);
      for(int j=edge_start; j < edge_end ; j++ ){
        int nd2 = this->graph.indices[j];
        sample.l1.nd2.push_back(nd2);
        sample.l1.edges.push_back(std::make_pair(nd1,nd2));
      }
    }
    sample.l1.remove_duplicates();
    int nodes_l1 = sample.l1.nd2.size();
    for(int i=0;i<nodes_l1;i++){
      int nd1 = sample.l1.nd2[i];
      int edge_start = this->graph.indptr[nd1];
      int edge_end = this->graph.indptr[nd1+1];
      if(edge_end - edge_start > 0)sample.l2.nd1.push_back(nd1);
      for(int j=edge_start; j < edge_end ; j++ ){
        int nd2 = this->graph.indices[j];
        sample.l2.nd2.push_back(nd2);
        sample.l2.edges.push_back(std::make_pair(nd1,nd2));
      }
    }

    sample.l2.remove_duplicates();
    // create csr

    sample.l1.create_csr();
    sample.l2.create_csr();
    this->fill_batch_data(sample.l2.nd2.data(),sample.l2.in_nodes, sample.l1.nd1.data(), sample.l1.out_nodes);
  }



  int number_of_batches(){
    return ((this->no_nodes - 1)/minibatch_size  + 1);
  }
};
