#pragma once

#include <vector>
#include "graph/sample.h"
#include "graph/dataset.h"
#include "util/duplicate.h"
#include <memory>
#include <random>
class NeighbourSampler{

  std::shared_ptr<Dataset> dataset;

  DuplicateRemover *dr;

  std::mt19937 random_number_engine;
  // Use constant fan out for all layers.
  int fanout=10;

  bool deterministic = false;
public:

  NeighbourSampler(std::shared_ptr<Dataset> dataset, int fanout, bool deterministic){
      this->dataset = dataset;
      this->fanout = fanout;
      this->deterministic = deterministic;
      dr = new DuplicateRemover(this->dataset->num_nodes);
  }

  void sample(std::vector<long> &target_nodes, Sample &s);

  void layer_sample(std::vector<long> &in, std::vector<long> &in_degrees, \
     std::vector<long> &offsets,
      std::vector<long> &indices);
};
