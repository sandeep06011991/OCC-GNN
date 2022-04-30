#pragma once
#include "util/conqueue.h"
#include "sample.h"
#include <vector>

class WorkerPool{

// Starts workers and puts all the work into the sampling pool.
// work queue.
  long num_nodes;
  int num_epochs;
  int minibatch_size;
  int num_workers;
  long * training_nodes;
  int num_batches;
  std::vector<int> workload_map;
  std::vector<int> storage_map[4];

public:
  ConQueue<Sample *> *generated_samples;
  ConQueue<std::vector<long> *> *work_queue;

  WorkerPool(long num_nodes, int num_epochs,
      int minibatch_size, int num_workers, Dataset *dataset,
        std::vector<int>& workload_map,
          std::vector<int>& storage_map[4],
          int batch_size);
          
  void run();

  void get_sample();

};
