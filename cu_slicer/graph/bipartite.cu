#include "bipartite.h"
#include "../util/cuda_utils.h"
#include "../util/duplicate.h"
#include "nvtx3/nvToolsExt.h"
#include <iostream>
// Replace everything with one local ordering.
// void BiPartite::reorder_local(cuslicer::DuplicateRemover *dr){
//   std::cout << "Not implemented\n";
//   std::cout << "Ideally everything in this function is sorted, only accts to merge\n";
// }

using namespace cuslicer;

void BiPartite::reorder_local(DuplicateRemover *dr){
  // dr->clear();
  // dr->order(in_nodes);
  // dr->replace(indices_L);
  // dr->replace(indices_R);
}
