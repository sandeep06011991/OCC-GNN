#include <string>
#ifndef DATASET_H
#define DATASET_H
class Dataset{

private:
  void read_meta_file();
  void read_node_data();
public:

  std::string BIN_DIR;
  // Meta-variables
  int num_nodes;
  int num_edges;
  int noClasses;
  int fsize;

  // data
  float *features;
  int *labels;

  // training splits.
  int * train_idx;
  int * test_idx;
  int train_idx_sizes;
  int test_idx_sizes;

  // check sum
  int csum_features;
  int csum_labels;

  Dataset(std::string dir);
};

#endif
