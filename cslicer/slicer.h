#pragma once

#include "dataset.h"
#include <vector>
#include "pybipartite.h"
#include <stdlib.h>
#include <algorithm>
#include <random>
#include <functional>
#include "util/conqueue.h"

class Slicer{

private:
  Dataset *dataset;
  std::vector<int> *storage_map[4];
  std::vector<int> * workload;
  int batch_size = 0;
  Sample sample;
  long * target_nodes;
  DuplicateRemover *dr;
  DuplicateRemover *out_dr;
  long num_nodes = 0;

  // Utilities
  vector<long> in;
  vector<long> out;
  vector<long> nbs;
  vector<long> neighbors;
  // std::mt19937 random_number_engine;
  // std::mt19937 gen;
 // std::uniform_int_distribution<> rng_coin(0, 10000);
  std::mt19937 random_number_engine;
  std::mt19937 gen;
  std::uniform_int_distribution<> rng_coin;
  ConQueue<PySample *> * generated_samples;
  ConQueue<std::vector<long> *> * work_queue;
  // std::uniform_int_distribution<> random_number_engine(0, 10000);
  // std::uniform_int_distribution<> rng_coin(0, 10000);
public:
  Slicer(Dataset * dataset, std::vector<int> *workload_map,
        std::vector<int>** storage_map,
          int batch_size, ConQueue<PySample *> * generated_samples,
            ConQueue<std::vector<long> *> * work_queue){
      std::cout << "cslicer 1\n";
      this->dataset = dataset;
      std::cout << "cslicer " << dataset->num_nodes <<"\n";
      this->num_nodes = dataset->num_nodes;
      std::cout << "cslicer 2\n";
      for(int i=0;i<4;i++){
        std::cout << "cslicer 2\n";
        this->storage_map[i] = storage_map[i];
      }
      std::cout << "cslicer 2\n";
      this->workload = workload_map;
      this->batch_size = batch_size;
      std::cout << "cslicer 2\n";
      this->dr = new DuplicateRemover(dataset->num_nodes);
      this->out_dr = new DuplicateRemover(dataset->num_nodes);
      this->target_nodes = (long *)malloc(sizeof(long) * dataset->num_nodes);
      for(long i=0;i<this->num_nodes;i++){
        this->target_nodes[i] = i;
      }
      std::cout << "cslicer 3\n";
      rng_coin = *(new std::uniform_int_distribution<>(0,10000));
      this->generated_samples = generated_samples;
      this->work_queue = work_queue;
      std::cout << "cslicer out\n";

  }

  int get_number_of_batches(){
    return ((this->num_nodes - 1)/batch_size  + 1);
  }

  void shuffle(){
    std::random_shuffle(&this->target_nodes[0], &this->target_nodes[this->num_nodes]);
  }

  void get_sample(vector<long> &batch);

  void neighbour_sample(long nd1, vector<long> &neighbours);

  void slice_layer(vector<long>& in, vector<long>& out, Layer& l);

  void simple_3_hop_sample(int batch_id);

  void run();

  void clear();
};
