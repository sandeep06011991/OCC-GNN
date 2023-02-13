#pragma once
#include <vector>
#include <cassert>
#include <unordered_map>
#include <iostream>
#include "vector.h"
#include <cub/cub.cuh>

void test_duplicate();

namespace cuslicer{
  // Used for all set operations in sampling
  // such as reordering vertex ids
  // removing seen vertives
  // TODO: Can make this static as only function runs at a time and there are no local properties to take advantage off.
  // This will minimize memory usage.
  
  class DuplicateRemover{

  public:
    // Temporar fix. Make private later
    DuplicateRemover(){}

    virtual void order(cuslicer::vector<long> &nodes) = 0;

    virtual cuslicer::vector<long>& get_used_nodes()  = 0;

    virtual ~DuplicateRemover() {}

    virtual void clear() = 0;

    virtual void replace(cuslicer::vector<long>& v) = 0;

    virtual void remove_nodes_seen(cuslicer::vector<long> &nodes) = 0;
  };

  //  DGL uses a hashmap.
  //  Better for scalability.
  //  Come back to this if its a problem
  class  ArrayMap: public  DuplicateRemover {
    cuslicer::vector<long> _tv;
    cuslicer::vector<long> _tv1;
    cuslicer::vector<long> _tv2;
    void * _df;
    int * mask;
    long mask_size;
    cuslicer::vector<long> used_nodes;
    void assert_no_duplicates(cuslicer::vector<long>  &nodes);

  public:


    ArrayMap(long num_nodes);

    cuslicer::vector<long>& get_used_nodes() ;
    // void order_and_remove_duplicates(thrust::device_vector<long>& nodes);

    void order(cuslicer::vector<long>& nodes);

    ~ArrayMap();

    void clear();

    void replace(cuslicer::vector<long>& v);

    void remove_nodes_seen(cuslicer::vector<long> &nodes);
  };

  class HashMap: public DuplicateRemover{
      // Plan for when stressed with memory.
      // Either use a simple hashing scheme or use HashTable of CUDA
  };

}

void test_duplicate();
