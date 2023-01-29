#pragma once
#include <vector>
#include <cassert>
#include <unordered_map>
#include <iostream>
#include <thrust/device_vector.h>
using namespace std;


// A Simple way to remove duplicates without using sorting or hashmaps.
// Only works as the max node size is already known and is finite
// Trading space for efficiency.
class DuplicateRemover{

public:
  // Temporar fix. Make private later
  DuplicateRemover(){}


  virtual void order(thrust::device_vector<long> &nodes) = 0;

  virtual thrust::device_vector<long>& get_used_nodes()  = 0;

  virtual ~DuplicateRemover() {}

  virtual void clear() = 0;

  virtual void replace(thrust::device_vector<long>& v) = 0;

  virtual void remove_nodes_seen(thrust::device_vector<long> &nodes) = 0;
};

class  ArrayMap: public  DuplicateRemover {
  thrust::device_vector<long> _tv;
  thrust::device_vector<long> _tv1;
  thrust::device_vector<long> _tv2;
  void * _df;
  int * mask;
  long mask_size;
  thrust::device_vector<long> used_nodes;
  void assert_no_duplicates(thrust::device_vector<long> &nodes);
public:


  ArrayMap(long num_nodes);

  thrust::device_vector<long>& get_used_nodes() ;
  // void order_and_remove_duplicates(thrust::device_vector<long>& nodes);

  void order(thrust::device_vector<long> &nodes);

  ~ArrayMap();

  void clear();

  void replace(thrust::device_vector<long>& v);

  void remove_nodes_seen(thrust::device_vector<long> &nodes);
};

void test_duplicate();
