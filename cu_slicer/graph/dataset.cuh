// Does only one job, Read the files from binary into memory.
// Data structures are not created here
// As placement decisions are taken here.
// Therefore contains some redundant data creation
#pragma once
#include <thrust/device_vector.h>
#include <string>
#ifndef DATASET_H
#define DATASET_H
class Dataset{

private:
  void read_meta_file();
  void read_node_data();
  void read_graph();
public:

  std::string BIN_DIR;
  // Meta-variables
  long num_nodes;
  long num_edges;
  int noClasses;
  int fsize;

  // data
  float *features;
  int *labels;

  // gpu_partition_map
  thrust::device_vector<int> partition_map;

  // training splits.
  long * train_idx;
  long * test_idx;
  long train_idx_sizes;
  long test_idx_sizes;

  // graph data.
  // Assume in node range same as out node range.
  long * indptr; // size = num_nodes + 1
  long * indices; // size = num_edges


  // check sum
  long csum_features;
  long csum_labels;
  long csum_offsets;
  long csum_edges;
  bool testing = true;
  Dataset(std::string dir, bool testing);


};

#endif
