#include "gtest/gtest.h"
#include "../../util/device_vector.h"
#include "../../util/duplicate.h"
#include "../../util/cub.h"
#include <stdlib.h>
#include <algorithm>
#include <numeric>
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

// TEST(ARRAYMAP, initial){
//   cudaSetDevice(0);
//   for(int i=0; i<100; i++)
//   {
//     std::vector<long> v(i);
//     std::generate(v.begin(), v.end(), std::rand);
//     std::vector<long> ref(i);
//     std::iota(ref.begin(), ref.end(), 0);
//     // ArrayMap * map = new ArrayMap(2000);
//     // device_vector<long> c(v);
//     // map->order(c);
//     // map->replace(c);
//     // EXPECT_TRUE(c.is_same(ref));
//     // map->clear();
//     // cuslicer::transform::cleanup();
//   }
// }

TEST(ARRAYMAP, order_test1){
  cudaSetDevice(0);
  std::vector<long> a = {500};
  ArrayMap * map = new ArrayMap(2000);
  device_vector<long> c(a);
  map->order(c);
  map->replace(c);
  std::vector<long> ref = {0};
  EXPECT_TRUE(c.is_same(ref));
  std::vector<long> d = {10,28,38,86,56,500};
  device_vector<long> e(d);
  map->order(e);
  map->replace(e);
  std::vector<long> ref1 = {1,2,3,4,5,0};
  EXPECT_TRUE(e.is_same(ref1));
  std::vector<long> f = {10,30,28,30,38,30,86,56,500};
  device_vector<long> g(f);
  map->order(g);
  map->replace(g);
  std::vector<long> ref2 = {1,6,2,6,3,6,4,5,0};
  EXPECT_TRUE(g.is_same(ref2));
  map->clear();
  cuslicer::transform::cleanup();
}

TEST(ARRAYMAP, order_test2){
  cudaSetDevice(0);
  std::vector<long> a = {200};
  ArrayMap * map = new ArrayMap(2000);
  device_vector<long> c(a);
  map->order(c);
  map->replace(c);
  std::vector<long> ref = {0};
  EXPECT_TRUE(c.is_same(ref));
  std::vector<long> d = {10,28,200,86,56,200};
  device_vector<long> e(d);
  map->order(e);
  map->replace(e);
  std::vector<long> ref1 = {1,2,0,3,4,0};
  EXPECT_TRUE(e.is_same(ref1));
  std::vector<long> f = {10,22,23,200,28,86,200,10};
  device_vector<long> g(f);
  map->order(g);
  map->replace(g);
  std::vector<long> ref2 = {1,5,6,0,2,3,0,1};
  EXPECT_TRUE(g.is_same(ref2));
  map->clear();
  cuslicer::transform::cleanup();
}

TEST(ARRAYMAP, order_test3){
  cudaSetDevice(0);
  std::vector<long> a = {0};
  ArrayMap * map = new ArrayMap(2000);
  device_vector<long> c(a);
  map->order(c);
  map->replace(c);
  std::vector<long> ref = {0};
  EXPECT_TRUE(c.is_same(ref));
  std::vector<long> d = {0,1,2,6,0,1,2};
  device_vector<long> e(d);
  map->order(e);
  map->replace(e);
  // due to no err checking of input
  // map has 1:1 2:2 as wel as 1:4 2:5
  std::vector<long> ref1 = {0,1,2,3,0,1,2};
  EXPECT_TRUE(e.is_same(ref1));
  std::vector<long> f = {6,7,12,9,1};
  device_vector<long> g(f);
  map->order(g);
  map->replace(g);
  std::vector<long> ref2 = {3,6,7,8,1};
  EXPECT_TRUE(g.is_same(ref2));
  map->clear();
  cuslicer::transform::cleanup();
}

TEST(ARRAYMAP, memory_test1){
  std::vector<long> a = {10,28,38,86,56,500};
  std::vector<long> correct = {10,30,28,30,38,30,86,56,500};
  device_vector<long> d_ref(correct);
  device_vector<long> d_vec(a);
  ArrayMap * map = new ArrayMap(2000);
  // device_vector<long> c(a);
  map->order(d_vec);
  a.insert(a.end(), correct.begin(), correct.end());
  device_vector<long> d(a);
  map->remove_nodes_seen(d);
  std::vector<long> correct1 = {30,30,30};
  EXPECT_TRUE(d.is_same(correct1));
  map->order(d_ref);
  std::vector<long> correct2 = {10,30,28,38,86,56,500};
  device_vector<long> f(correct2);
  map->remove_nodes_seen(f);
  for(int i = 0; i < f.size(); i++){
    std::cout << f[i] << " ";
  }
  std::vector<long> cc = {};
  EXPECT_TRUE(f.is_same(cc));
  map->clear();
  cuslicer::transform::cleanup();
}