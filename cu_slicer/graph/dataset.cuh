// Does only one job, Read the files from binary into memory.
// Data structures are not created here
// As placement decisions are taken here.
// Therefore contains some redundant data creation
#pragma once
#include <string>
#include "../util/cuda_utils.h"
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
  // const float *features;
  // const int *labels;

  // gpu_partition_map
  int * partition_map_d;

  // graph data.
  // Assume in node range same as out node range.
  long * indptr_d; // size = num_nodes + 1
  long * indices_d; // size = num_edges


  // check sum
  long csum_features;
  long csum_labels;
  long csum_offsets;
  long csum_edges;
  bool testing = true;
  Dataset(std::string dir, bool testing);

  ~Dataset(){
    gpuErrchk(cudaFree(indptr_d));
    gpuErrchk(cudaFree(indices_d));
  }
};

#endif
