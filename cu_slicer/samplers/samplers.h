#pragma once

#include <vector>
#include "graph/sample.h"
#include "graph/dataset.h"
#include "util/duplicate.h"
#include <memory>
#include <random>
#include <thrust/device_vector.h>
class NeighbourSampler{

  std::shared_ptr<Dataset> dataset;

  DuplicateRemover *dr;

  std::mt19937 random_number_engine;
  // Use constant fan out for all layers.
  int fanout=10;

  bool deterministic = false;

  bool self_edge = false;
public:

  NeighbourSampler(std::shared_ptr<Dataset> dataset, int fanout, bool deterministic, bool self_edge){
      this->dataset = dataset;
      this->fanout = fanout;
      this->deterministic = deterministic;
      dr = new ArrayMap(this->dataset->num_nodes);
      this->self_edge = self_edge ;
  }

  void sample(thrust::device_vector<long> &target_nodes, Sample &s);
private:
  void layer_sample(thrust::device_vector<long> &in, thrust::device_vector<long> &in_degrees, \
     thrust::device_vector<long> &offsets,
     thrust::device_vector<long> &indices);
};
