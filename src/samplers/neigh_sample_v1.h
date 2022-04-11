#pragma once
#include <algorithm>
#include <assert.h>
#include <samplers/sample_v1.h>
#include <stdlib.h>
#include <util/timer.h>
#include <cstring>
#include "nvToolsExt.h"
// creates minibatches of given size by shuffling target node
// with full 2 hop neighbourhood.
// Fixme:: Move the graph class to different position.
struct Graph{
  long num_nodes;
  long num_edges;
  long *indptr;
  long *indices;
};

class NeighSampler{

  Graph graph;
  // Full graph
  long no_nodes;
  long no_edges;
  // Instantitate from sample 1-n

  // Features of the last sampled_hop
  float * full_features;
  int * full_labels;

  // Utilities which support minibatching.
  // int next_minibatch;
  int minibatch_size;
  int current_minibatch_size;
  long * target_nodes;
  int fsize;
  int layers;
  int * reorder_index;

public:
  TwoHopSample sample;
  float * batch_features = nullptr;
  int * batch_labels = nullptr;

  NeighSampler(long no_nodes, long no_edges, long *ind_ptr, long *indices,
      int max_batch_size, float *features, int * labels, int fsize, int layers){
    this->graph.num_nodes = no_nodes;
    this->graph.num_edges = no_edges;
    this->graph.indptr = ind_ptr;
    this->graph.indices = indices;
    this->no_nodes = no_nodes;
    this->no_edges = no_edges;
    this->target_nodes = (long *)malloc(sizeof(long) * no_nodes);
    this->full_features = features;
    this->full_labels = labels;
    for(int i =0;i<no_nodes;i++){
      this->target_nodes[i] = i;
    }
    this->minibatch_size = max_batch_size;
    this->fsize = fsize;

    this->reorder_index = (int *)malloc(sizeof(int) * no_nodes);
    memset(this->reorder_index,0,sizeof(int) * no_nodes);
    shuffle();
  }

  void shuffle(){
    std::random_shuffle(&this->target_nodes[0], &this->target_nodes[this->no_nodes]);
  }

  void fill_batch_data(long* in_nodes, int no_in, long* out_nodes, int no_out){
    if(batch_features != nullptr){
      free(batch_features);
    }
    if(batch_labels != nullptr){
      free(batch_labels);
    }
    batch_features = (float *)malloc(sizeof(float)* no_in * this->fsize);
    batch_labels = (int *)malloc(sizeof(int) * no_out);
    start_timer(SAMPL_F_DATA_FORMAT);
    #pragma omp parallel for
    for(int i=0; i < no_in;i++){
      memcpy(&batch_features[i*this->fsize],
          &full_features[in_nodes[i]*this->fsize],sizeof(float) * this->fsize);
      // for(int j=0;j < this->fsize; j++){
      //   batch_features[i*this->fsize+j]= (full_features[in_nodes[i] * this->fsize + j]);
      // }
    }

    for(int i=0; i< no_out; i++){
      batch_labels[i] = full_labels[out_nodes[i]];
    }
    stop_timer(SAMPL_F_DATA_FORMAT);
  }


  void sample_one_layer(SampleLayer &sample, long *tgt, int no_nodes){
    start_timer(SAMPLE_CREATION);
    // Create csr without reordering.
    sample.clear();
    for(int i=0;i<no_nodes;i++){
      long nd1 = tgt[i];
      long edge_start = this->graph.indptr[nd1];
      long edge_end = this->graph.indptr[nd1+1];
      if(edge_end - edge_start>0) sample.nd1.push_back(nd1);
      long no_neighbours = edge_end - edge_start;
      if(no_neighbours < 10){
        sample.indptr.push_back(no_neighbours);
        for(long j=edge_start; j < edge_end ; j++ ){
          long nd2 = this->graph.indices[j];
          sample.indices.push_back(nd2);
        }
      }else{
        sample.indptr.push_back(10);
        for(int j=0;j<10;j++){
          int rand_nb = rand()%no_neighbours;
          long nd2 = this->graph.indices[edge_start+rand_nb];
          sample.indices.push_back(nd2);
        }
      }
    }
    stop_timer(SAMPLE_CREATION);
    start_timer(DUPLICATE_LAYER);
    sample.reorder(this->reorder_index );
    stop_timer(DUPLICATE_LAYER);
  }

  void get_sample(int batchId){
    nvtxRangePushA("Sampling start");
    assert(batchId * this->minibatch_size < this->no_nodes);
    long * tgt = &this->target_nodes[this->minibatch_size * batchId];
    int no_nodes =  minibatch_size;
    if(this->minibatch_size * (batchId + 1) > this->no_nodes){
      no_nodes = this->no_nodes - (this->minibatch_size * batchId);
    }
    this->current_minibatch_size = no_nodes;
    start_timer(SAMPLE_CREATION);
    // Sample 2-hop neighbourhoods.
    // sample first layer
    sample_one_layer(sample.l1, tgt, no_nodes);
    sample_one_layer(sample.l2, sample.l1.nd2.data(), sample.l2.nd2.size());
    stop_timer(SAMPLE_CREATION);

    start_timer(FILL_DATA);
    this->fill_batch_data(sample.l2.nd2.data(),sample.l2.in_nodes, sample.l1.nd1.data(), sample.l1.out_nodes);
    stop_timer(FILL_DATA);
    nvtxRangePop();
  }


  void debug(){
    // auto l1 = sample.l1.nd1;
    // auto l2 = sample.l1.nd2;
    // auto l3 = sample.l2.nd2;
    // std::cout << "Layer 1\n";
    // for(int i=0;i<l1.size();i++){
    //   std::cout << l1[i] <<" ";
    // }
    // std::cout << "\n";
    // std::cout << "Layer 2\n";
    // for(int i=0;i<l2.size();i++){
    //   std::cout << l2[i] <<" ";
    // }
    //
    // std::cout << "\n";
    // std::cout << "Layer 3\n";
    // for(int i=0;i<l3.size();i++){
    //   std::cout << l3[i] <<" ";
    // }
    //
    // std::cout << "\n";
  }
  int number_of_batches(){
    return ((this->no_nodes - 1)/minibatch_size  + 1);
  }
};
