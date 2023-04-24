#pragma once
#include <vector>
#include <cassert>
#include <unordered_map>
#include <iostream>
#include "device_vector.h"
#include <cub/cub.cuh>
#include "types.h"

void test_duplicate();

namespace cuslicer{
  // Used for all set operations in sampling
  // such as reordering vertex ids
  // removing seen vertives
  // TODO: Can make this static as only function runs at a time and there are no local properties to take advantage off.
  // This will minimize memory usage.
  // If this map ends up being try with dgl hash map. 
   
  class DuplicateRemover{

  public:
    // Temporar fix. Make private later
    DuplicateRemover(){}

    virtual void order(device_vector<long> &nodes) = 0;

    virtual device_vector<long>& get_used_nodes()  = 0;

    virtual ~DuplicateRemover() {}

    virtual void clear() = 0;

    virtual void replace(device_vector<long>& v) = 0;

    virtual void remove_nodes_seen(device_vector<long> &nodes) = 0;
  };

  //  DGL uses a hashmap.
  //  Better for scalability.
  //  Come back to this if its a problem
  class  ArrayMap: public  DuplicateRemover {
    device_vector<NDTYPE> _tv;
    device_vector<NDTYPE> _tv1;
    device_vector<NDTYPE> _tv2;

    int * mask = nullptr;
    long mask_size = 0;
    device_vector<NDTYPE> used_nodes;
    void assert_no_duplicates(device_vector<NDTYPE>  &nodes);

  public:


    ArrayMap(NDTYPE num_nodes);

    device_vector<long>& get_used_nodes() ;
    // void order_and_remove_duplicates(thrust::device_vector<long>& nodes);

    void order(device_vector<long>& nodes);

    ~ArrayMap();

    void clear();

    void replace(device_vector<long>& v);

    void remove_nodes_seen(device_vector<long> &nodes);
  };

  class HashMap: public DuplicateRemover{
      // Plan for when stressed with memory.
      // Either use a simple hashing scheme or use HashTable of CUDA
  };

}

void test_duplicate();
