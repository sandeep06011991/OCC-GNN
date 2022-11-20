#pragma once
#include <vector>
#include <cassert>
#include <unordered_map>
#include <iostream>
using namespace std;


// A Simple way to remove duplicates without using sorting or hashmaps.
// Only works as the max node size is already known and is finite
// Trading space for efficiency.
class DuplicateRemover{

public:
  // Temporar fix. Make private later
  DuplicateRemover(){}

  virtual void order_and_remove_duplicates(std::vector<long>& nodes) = 0;

  virtual void order(std::vector<long> &nodes) = 0;


  virtual ~DuplicateRemover() {}

  virtual void clear() = 0;

  virtual void replace(std::vector<long>& v) = 0;
};

class  ArrayMap: public  DuplicateRemover {

  std::vector<long> used_nodes;
  int * mask;
public:
  ArrayMap(long num_nodes);

  void order_and_remove_duplicates(std::vector<long>& nodes);

 void order(std::vector<long> &nodes);

  ~ArrayMap();

 void clear();

 void replace(std::vector<long>& v);
};

class HashMap: public DuplicateRemover{

  std::unordered_map<long,long> map;
  int count;
public:
  HashMap(long num_nodes);

  void order_and_remove_duplicates(std::vector<long>& nodes);

 void order(std::vector<long> &nodes);

  ~HashMap(){}

 void clear();

 void replace(std::vector<long>& v);
};
