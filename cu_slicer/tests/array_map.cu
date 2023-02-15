#include "gtest/gtest.h"
#include "../util/device_vector.h"
#include "../util/duplicate.h"
#include "../util/cub.h"
using namespace cuslicer;

TEST(ARRAYMAP, basic){
  cudaSetDevice(0);
  std::vector<long> a = { 14, 12, 9};
  device_vector<long> d_vec(a);
  ArrayMap * map = new ArrayMap(2000);
  device_vector<long> c(a);
  map->order(d_vec);
  map->replace(c);
  std::vector<long> ref = {0,1,2};
  EXPECT_TRUE(c.is_same(ref));
  std::vector<long> d = {12,15,6};
  device_vector<long> e(d);
  map->order(e);
  map->replace(e);
  std::vector<long> ref1 = {1,3,4};
  EXPECT_TRUE(e.is_same(ref1));
  map->clear();
  cuslicer::transform::cleanup();
}

// Test memory cleanup.
TEST(ARRAYMAP, memory){
  cudaSetDevice(0);
  std::vector<long> a = { 14, 12, 9};
  device_vector<long> d_vec(a);
  ArrayMap * map = new ArrayMap(2000);
  device_vector<long> c(a);
  map->order(d_vec);
  device_vector<long>  us = map->get_used_nodes();
  device_vector<long>  us1 = map->get_used_nodes();
  device_vector<long>  us2 = map->get_used_nodes();
  device_vector<long>  us3 = map->get_used_nodes();
  map->clear();
  cuslicer::transform::cleanup();
}
