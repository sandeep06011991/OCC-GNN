#include<iostream>
#include "graph/dataset.cuh"
#include "util/environment.h"
#include "memory"
#include <vector>
// #include "tests/test.h"
#include "util/device_vector.h"
#include "util/cub.h"
#include "util/duplicate.h"
#include <chrono>
#include <iostream>
#include "util/traversal.h"
using namespace std;
using namespace std::chrono;

int main(){
// Test1: Read graph datastructure.
// test_duplicate();
  cudaSetDevice(0);
// std::cout << "hello world\n"; 
  // std::string graph_name = "synth_8_2";
  std::string graph_name = "ogbn-products";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false);
  traversal(dataset);
  std::cout << "Hello world\n";
}
