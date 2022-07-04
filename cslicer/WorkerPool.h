#pragma once
#include "util/conqueue.h"
#include "pybipartite.h"
#include <vector>
#include "dataset.h"
#include "slicer.h"


class WorkerPool{

// Starts workers and puts all the work into the sampling pool.
// work queue.
  long num_nodes;
  int num_epochs;
  int minibatch_size;
  int num_workers;
  long * training_nodes;
  int num_batches;
  std::vector<int>  *workload_map;
  std::vector<int>  *storage_map[4];
  int gpu_capacity[4];
  Slicer ** samplers;
  Dataset *dataset;
  std::thread **th;
public:
  ConQueue<PySample *> *generated_samples;
  ConQueue<std::vector<long> *> *work_queue;

  WorkerPool(long num_nodes, int num_epochs,
      int minibatch_size, int num_workers,
      Dataset *dataset,
        std::vector<int>* workload_map,
          std::vector<int>** storage_map,
           int gpu_capacity[4]
          );

  void run();

  void get_sample();

  PySample * pop_object(){
    std::cout << "pop object \n";
    PySample * sample =  this->generated_samples->pop_object();
    std::cout << "pop object complete\n";
    return sample;
  }
};
