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

  bool self_edge = false;
public:

  NeighbourSampler(std::shared_ptr<Dataset> dataset, int fanout, bool deterministic, bool self_edge){
      this->dataset = dataset;
      this->fanout = fanout;
      this->deterministic = deterministic;
      dr = new ArrayMap(this->dataset->num_nodes);
      this->self_edge = self_edge ;
  }

  void sample(std::vector<long> &target_nodes, Sample &s);
private:
  void layer_sample(std::vector<long> &in, std::vector<long> &in_degrees, \
     std::vector<long> &offsets,
      std::vector<long> &indices);
};
