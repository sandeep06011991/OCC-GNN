
#include "graph/sample.h"
#include "graph/sliced_sample.h"

void test_sample_partition_consistency(Sample &s,
        PartitionedSample &p,
      std::vector<int> storage_map[4], int gpu_capacity[4], int num_nodes);
