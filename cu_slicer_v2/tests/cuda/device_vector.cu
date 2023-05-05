#include "gtest/gtest.h"
#include "../../util/device_vector.h"
#include "../../util/duplicate.h"
#include "../../util/cub.h"
using namespace cuslicer;

TEST(VECTOR, copy){
  cudaSetDevice(0);
  std::vector<long> _a = {1,2,3,4};
  cuslicer::device_vector<long> b(_a);
  cuslicer::device_vector<long> c;
  c = b;
  c.set_value(0,3);
  std::vector<long> ref = {3,2,3,4};
  EXPECT_TRUE(b.is_same(_a));
  EXPECT_TRUE(c.is_same(ref));
  cuslicer::transform::cleanup();
}
