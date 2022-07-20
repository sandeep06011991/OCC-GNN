#pragma once
#include <vector>
#include <cassert>

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
