#include "graph/sample.h"

// Walk and modify the constructed sample.
// This allows us to add redundant computation and further decrease the shuffle cost.
// Constructed samples can be transformed before and after they are slicing
// Similar to a compiler pass.
void walk(Sample &s, int *workload_map, int * storage_map);
