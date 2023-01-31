#pragma once

#include <vector>
#include "graph/sample.h"
#include "graph/dataset.cuh"
#include "util/duplicate.h"
#include <memory>
#include <random>
#include <thrust/device_vector.h>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

class NeighbourSampler{

  std::shared_ptr<Dataset> dataset;

  DuplicateRemover *dr;
  thrust::device_vector<long> _t;
  std::mt19937 random_number_engine;
  // Use constant fan out for all layers.

  bool deterministic = false;
  std::vector<int> fanout;
  bool self_edge = false;
  curandState* dev_curand_states;
  const int TOTAL_RAND_STATES = MAX_BLOCKS * THREAD_SIZE;
public:

  NeighbourSampler(std::shared_ptr<Dataset> dataset,
      vector<int> fanout, bool self_edge);

  void sample(thrust::device_vector<long> &target_nodes, Sample &s);

private:
  void layer_sample(thrust::device_vector<long> &in, \
      thrust::device_vector<long> &in_degrees, \
        thrust::device_vector<long> &offsets,
        thrust::device_vector<long> &indices, int fanout);
};
