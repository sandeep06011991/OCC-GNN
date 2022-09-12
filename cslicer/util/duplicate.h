#pragma once
#include <vector>
#include <cassert>

// A Simple way to remove duplicates without using sorting or hashmaps.
// Only works as the max node size is already known and is finite
// Trading space for efficiency. 
class DuplicateRemover{

public:
  // Temporar fix. Make private later
  std::vector<long> used_nodes;
  int * mask;

  DuplicateRemover(long num_nodes);

  void order_and_remove_duplicates(std::vector<long>& nodes);

  ~ DuplicateRemover();

  void clear();

  void replace(std::vector<long>& v);
};
