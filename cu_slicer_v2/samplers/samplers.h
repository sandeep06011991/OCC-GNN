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
#include "../util/types.h"

using namespace cuslicer;

class NeighbourSampler{

  std::shared_ptr<Dataset> dataset;

  DuplicateRemover *dr;
  cuslicer::device_vector<NDTYPE> _t;
  std::mt19937 random_number_engine;
  // Use constant fan out for all layers.

  bool deterministic = false;
  std::vector<int> fanout;
  bool self_edge = false;
  
  cuslicer::device_vector<NDTYPE> _t1;
public:
  curandState*  dev_curand_states;
  const int TOTAL_RAND_STATES = MAX_BLOCKS * BLOCK_SIZE;

  NeighbourSampler(std::shared_ptr<Dataset> dataset,
      vector<int> fanout, bool self_edge);

  void sample(device_vector<NDTYPE> &target_nodes, Sample &s);

private:
  void layer_sample(device_vector<NDTYPE> &in, \
      device_vector<NDTYPE> &in_degrees, \
        device_vector<NDTYPE> &offsets,
        device_vector<NDTYPE> &indices, int fanout);
};
