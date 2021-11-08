// Does only one job, Read the files from binary into memory.
// Data structures are not created here
// As placement decisions are taken here.
// Therefore contains some redundant data creation

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

  // graph data.
  // Assume in node range same as out node range.
  int * indptr; // size = num_nodes + 1
  int * indices; // size = num_edges

  // check sum
  int csum_features;
  int csum_labels;
  int csum_offsets;
  int csum_edges;

  Dataset(std::string dir);
};

#endif
