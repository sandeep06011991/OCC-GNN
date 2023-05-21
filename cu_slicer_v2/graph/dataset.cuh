// Does only one job, Read the files from binary into memory.
// Data structures are not created here
// As placement decisions are taken here.
// Therefore contains some redundant data creation
#pragma once
#include <string>
#include "../util/cuda_utils.h"
#include "../util/device_vector.h"
#include "../util/types.h"

#ifndef DATASET_H
#define DATASET_H

namespace cuslicer{

class Dataset{

private:
  void read_meta_file();
  void read_node_data();
  void read_graph();
public:

  std::string BIN_DIR;
  // Meta-variables
  NDTYPE num_nodes;
  NDTYPE num_edges;
  int noClasses;
  int fsize;

  // data
  // const float *features;
  // const int *labels;

  // gpu_partition_map
  cuslicer::device_vector<PARTITIONIDX> partition_map_d;

  // graph data.
  NDTYPE * indptr_h;
  NDTYPE * indices_h;
  // Assume in node range same as out node range.
  NDTYPE * indptr_d; // size = num_nodes + 1
  NDTYPE * indices_d; // size = num_edges


  // check sum
  long csum_features;
  long csum_labels;
  long csum_offsets;
  long csum_edges;
  bool testing = true;
  int num_partitions;
  bool random = false;
  bool UVA = false;
  Dataset(std::string dir, int num_partitions, bool random, bool UVA);

  ~Dataset(){
    if(this->UVA){
      gpuErrchk(cudaFreeHost(indptr_h));
      gpuErrchk(cudaFreeHost(indices_h));
    }else{
      gpuErrchk(cudaFree(indptr_d));
      gpuErrchk(cudaFree(indices_d));
    }
    // gpuErrchk(cudaFree(indptr_d));
    // gpuErrchk(cudaFree(indices_d));
  }
};

}
#endif
