#include "gtest/gtest.h"
#include "../util/device_vector.h"
#include "../util/cub.h"

TEST(CUB, reduce){
  cudaSetDevice(0);
  std::vector<long> a = {3,2,1,5,5,5};
  cuslicer::device_vector<long> b(a);
  EXPECT_EQ(21, cuslicer::transform::reduce(b));
  cuslicer::transform::cleanup();
  EXPECT_TRUE(true);
}

TEST(CUB, sort){
  cudaSetDevice(0);
  std::vector<long> a = {3,2,1,5,5,5};
  cuslicer::device_vector<long> b(a);
  cuslicer::device_vector<long> c;
  cuslicer::transform::sort(b,b);
  std::vector<long> ref = {1,2,3,5,5,5};
  EXPECT_TRUE(b.is_same(ref));
  cuslicer::transform::cleanup();
}

TEST(CUB, unique){
  cudaSetDevice(0);
  std::vector<long> a = {1,2,3,5,5,5,5};
  std::vector<long> ref = {1,2,3,5};
  cuslicer::device_vector<long> b(a);
  cuslicer::device_vector<long> c;
  cuslicer::transform::unique(b,c);
  EXPECT_TRUE(c.is_same(ref));
  cuslicer::transform::cleanup();
}

TEST(CUB, exclusive_scan){
  cudaSetDevice(0);
  std::vector<long> a = {1,2,3,5,5};
  std::vector<long> ref = {0,1,3,6,11};
  cuslicer::device_vector<long> b(a);
  cuslicer::device_vector<long> c;
  cuslicer::transform::exclusive_scan(b,c);
  EXPECT_TRUE(c.is_same(ref));
  cuslicer::transform::cleanup();
}

TEST(CUB, inclusive_scan_int ){
  cudaSetDevice(0);
  std::vector<int> a = {1,2,3,5,5};
  std::vector<int> ref = {1,3,6,11,16};
  cuslicer::device_vector<int> b(a);
  cuslicer::device_vector<int> c;
  cuslicer::transform:: self_inclusive_scan_int(b);
  EXPECT_TRUE(b.is_same(ref));
  cuslicer::transform::cleanup();
}
