#pragma once
#include<algorithm>
#include<assert.h>
#include<samplers/sample.h>
#include <stdlib.h>
#include <util/timer.h>
#include <cstring>
#include "nvToolsExt.h"
// creates minibatches of given size by shuffling target node
// with full 2 hop neighbourhood.
// Fixme:: Move the graph class to different position.
struct Graph{
  int num_nodes;
  int num_edges;
  int *indptr;
  int *indices;
};

class TwoHopNeighSampler{

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

  TwoHopNeighSampler(int no_nodes, int no_edges, int *ind_ptr, int *indices,
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


  void get_sample(int batchId){
    nvtxRangePushA("Sampling start");
    assert(batchId * this->minibatch_size < this->no_nodes);
    int * tgt = &this->target_nodes[this->minibatch_size * batchId];
    int no_nodes =  minibatch_size;
    if(this->minibatch_size * (batchId + 1) > this->no_nodes){
      no_nodes = this->no_nodes - (this->minibatch_size * batchId);
    }
    this->current_minibatch_size = no_nodes;
    // Sample 2-hop neighbourhoods.
    sample.clear();
    start_timer(SAMPLE_CREATION);
    // #pragma omp parallel for
    for(int i=0;i<no_nodes;i++){
      int nd1 = tgt[i];
      int edge_start = this->graph.indptr[nd1];
      int edge_end = this->graph.indptr[nd1+1];
      if(edge_end - edge_start>0)sample.l1.nd1.push_back(nd1);
      int no_neighbours = edge_end - edge_start;
      if(no_neighbours < 10){
        for(int j=edge_start; j < edge_end ; j++ ){
          int nd2 = this->graph.indices[j];
          sample.l1.nd2.push_back(nd2);
          sample.l1.edges.push_back(std::make_pair(nd1,nd2));
        }
      }else{
        for(int j=0;j<10;j++){
          int rand_nb = rand()%no_neighbours;
          int nd2 = this->graph.indices[edge_start+rand_nb];
          sample.l1.nd2.push_back(nd2);
          sample.l1.edges.push_back(std::make_pair(nd1,nd2));
        }
      }
    }
    stop_timer(SAMPLE_CREATION);
    start_timer(DUPLICATE_LAYER);
    sample.l1.remove_duplicates();
    stop_timer(DUPLICATE_LAYER);
    int nodes_l1 = sample.l1.nd2.size();
    // #pragma omp parallel for
    start_timer(SAMPLE_CREATION);
    for(int i=0;i<nodes_l1;i++){
      int nd1 = sample.l1.nd2[i];
      int edge_start = this->graph.indptr[nd1];
      int edge_end = this->graph.indptr[nd1+1];

      int no_neighbours = edge_end - edge_start;
      if(edge_end - edge_start > 0)sample.l2.nd1.push_back(nd1);
      if(no_neighbours < 10){
        for(int j=edge_start; j < edge_end ; j++ ){
          int nd2 = this->graph.indices[j];
          sample.l2.nd2.push_back(nd2);
          sample.l2.edges.push_back(std::make_pair(nd1,nd2));
        }
      }else{
        for(int j=0;j<10;j++){
          int rand_nb = rand()%no_neighbours;
          int nd2 = this->graph.indices[edge_start+rand_nb];
          sample.l2.nd2.push_back(nd2);
          sample.l2.edges.push_back(std::make_pair(nd1,nd2));
        }
      }
    }
    stop_timer(SAMPLE_CREATION);
    start_timer(DUPLICATE_LAYER);
    sample.l2.remove_duplicates();
    stop_timer(DUPLICATE_LAYER);
    // create csr
    start_timer(CREATE_CSR);
    sample.l1.create_csr();
    sample.l2.create_csr();
    stop_timer(CREATE_CSR);

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

class ThreeHopNeighSampler{

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
    ThreeHopSample sample;
    float * batch_features = nullptr;
    int * batch_labels = nullptr;

    ThreeHopNeighSampler(int no_nodes, int no_edges, int *ind_ptr, int *indices,
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


    void get_sample(int batchId){
      nvtxRangePushA("Sampling start");
      assert(batchId * this->minibatch_size < this->no_nodes);
      int * tgt = &this->target_nodes[this->minibatch_size * batchId];
      int no_nodes =  minibatch_size;
      if(this->minibatch_size * (batchId + 1) > this->no_nodes){
        no_nodes = this->no_nodes - (this->minibatch_size * batchId);
      }
      this->current_minibatch_size = no_nodes;
      // Sample 2-hop neighbourhoods.
      sample.clear();
      start_timer(SAMPLE_CREATION);
      // #pragma omp parallel for
      for(int i=0;i<no_nodes;i++){
        int nd1 = tgt[i];
        int edge_start = this->graph.indptr[nd1];
        int edge_end = this->graph.indptr[nd1+1];
        if(edge_end - edge_start>0)sample.l1.nd1.push_back(nd1);
        int no_neighbours = edge_end - edge_start;
        if(no_neighbours < 10){
          for(int j=edge_start; j < edge_end ; j++ ){
            int nd2 = this->graph.indices[j];
            sample.l1.nd2.push_back(nd2);
            sample.l1.edges.push_back(std::make_pair(nd1,nd2));
          }
        }else{
          for(int j=0;j<10;j++){
            int rand_nb = rand()%no_neighbours;
            int nd2 = this->graph.indices[edge_start+rand_nb];
            sample.l1.nd2.push_back(nd2);
            sample.l1.edges.push_back(std::make_pair(nd1,nd2));
          }
        }
      }
      stop_timer(SAMPLE_CREATION);
      start_timer(DUPLICATE_LAYER);
      sample.l1.remove_duplicates();
      stop_timer(DUPLICATE_LAYER);
      int nodes_l1 = sample.l1.nd2.size();
      // #pragma omp parallel for
      start_timer(SAMPLE_CREATION);
      for(int i=0;i<nodes_l1;i++){
        int nd1 = sample.l1.nd2[i];
        int edge_start = this->graph.indptr[nd1];
        int edge_end = this->graph.indptr[nd1+1];

        int no_neighbours = edge_end - edge_start;
        if(edge_end - edge_start > 0)sample.l2.nd1.push_back(nd1);
        if(no_neighbours < 10){
          for(int j=edge_start; j < edge_end ; j++ ){
            int nd2 = this->graph.indices[j];
            sample.l2.nd2.push_back(nd2);
            sample.l2.edges.push_back(std::make_pair(nd1,nd2));
          }
        }else{
          for(int j=0;j<10;j++){
            int rand_nb = rand()%no_neighbours;
            int nd2 = this->graph.indices[edge_start+rand_nb];
            sample.l2.nd2.push_back(nd2);
            sample.l2.edges.push_back(std::make_pair(nd1,nd2));
          }
        }
      }
      stop_timer(SAMPLE_CREATION);
      start_timer(DUPLICATE_LAYER);
      sample.l2.remove_duplicates();
      stop_timer(DUPLICATE_LAYER);
      int nodes_l2 = sample.l2.nd2.size();
      // #pragma omp parallel for
      start_timer(SAMPLE_CREATION);
      for(int i=0;i<nodes_l2;i++){
        int nd1 = sample.l2.nd2[i];
        int edge_start = this->graph.indptr[nd1];
        int edge_end = this->graph.indptr[nd1+1];

        int no_neighbours = edge_end - edge_start;
        if(edge_end - edge_start > 0)sample.l3.nd1.push_back(nd1);
        if(no_neighbours < 10){
          for(int j=edge_start; j < edge_end ; j++ ){
            int nd2 = this->graph.indices[j];
            sample.l3.nd2.push_back(nd2);
            sample.l3.edges.push_back(std::make_pair(nd1,nd2));
          }
        }else{
          for(int j=0;j<10;j++){
            int rand_nb = rand()%no_neighbours;
            int nd2 = this->graph.indices[edge_start+rand_nb];
            sample.l3.nd2.push_back(nd2);
            sample.l3.edges.push_back(std::make_pair(nd1,nd2));
          }
        }
      }
      stop_timer(SAMPLE_CREATION);
      start_timer(DUPLICATE_LAYER);
      sample.l3.remove_duplicates();
      stop_timer(DUPLICATE_LAYER);

      // create csr
      start_timer(CREATE_CSR);
      sample.l1.create_csr();
      sample.l2.create_csr();
      sample.l3.create_csr();
      stop_timer(CREATE_CSR);

      start_timer(FILL_DATA);
      this->fill_batch_data(sample.l3.nd2.data(),sample.l3.in_nodes, sample.l1.nd1.data(), sample.l1.out_nodes);
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
