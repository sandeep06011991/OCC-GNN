#pragma once

#include <vector>
#include "../graph/sample.h"
#include "../graph/dataset.cuh"
#include "../util/duplicate.h"
#include "../util/device_vector.h"
#include <memory>
#include <random>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

using namespace cuslicer;

class NeighbourSampler{

  std::shared_ptr<Dataset> dataset;

  DuplicateRemover *dr;
  cuslicer::device_vector<long> _t;
  std::mt19937 random_number_engine;
  // Use constant fan out for all layers.

  bool deterministic = false;
  std::vector<int> fanout;
  bool self_edge = false;
  curandState* dev_curand_states;
  const int TOTAL_RAND_STATES = MAX_BLOCKS * BLOCK_SIZE;
  cuslicer::device_vector<long> _t1;
public:

  NeighbourSampler(std::shared_ptr<Dataset> dataset,
      vector<int> fanout, bool self_edge);

  void sample(device_vector<long> &target_nodes, Sample &s);

private:
  void layer_sample(device_vector<long> &in, \
      device_vector<long> &in_degrees, \
        device_vector<long> &offsets,
        device_vector<long> &indices, int fanout);
};
