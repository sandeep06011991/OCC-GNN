#include "gtest/gtest.h"
#include "../../util/device_vector.h"
#include "../../util/duplicate.h"
#include "../../util/cub.h"
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
  std::vector<long> a;
  cudaSetDevice(0);
  for(int j = 0; j < 100; j++){
        a.push_back(j);
  }
  std::vector<long> correct;
  for(int i = 100; i < 200; i++){
    correct.push_back(i);
  }
  device_vector<long> d_ref(correct);
  device_vector<long> d_vec(a);
  ArrayMap * map = new ArrayMap(2000);
  device_vector<long> c(a);
  map->order(d_vec);
  a.insert(a.end(), correct.begin(), correct.end());
  device_vector<long> d(a);
  map->remove_nodes_seen(d);
  device_vector<long>  us = map->get_used_nodes();
  EXPECT_TRUE(d.is_same(correct));
  map->clear();
  cuslicer::transform::cleanup();
}
