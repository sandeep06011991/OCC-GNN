#include<iostream>
#include "../graph/dataset.cuh"
#include "../util/environment.h"
#include "memory"
#include "../graph/sample.h"
#include "../samplers/samplers.h"
#include <vector>
#include "../util/device_vector.h"
#include "../util/cub.h"
#include "../util/duplicate.h"
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;
#include "gtest/gtest.h"
using namespace cuslicer;

TEST(SAMPLING, perf){
  cudaSetDevice(0);
// std::cout << "hello world\n";
  // std::string graph_name = "synth_8_2";
  float TARGET_TIME = .006;
  std::string graph_name = "ogbn-arxiv";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false);
// std::cout << "Read synthetic dataset\n ";
// // // Test2: Construct simple k-hop neighbourhood sample.
// // // Sample datastructure.
  int num_layers = 3  ;
  Sample *s1  = new Sample(num_layers);
  vector<int> fanout({20,20,20});
  bool self_edge = false;
  std::vector<long> training_nodes;
  for(int i=0;i<dataset->num_nodes;i++){
      training_nodes.push_back(i);
  }

  int batch_size = 4096;
  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, self_edge);
  for(int i=0; i < dataset->num_nodes - batch_size; i = i + batch_size){
    std::vector<long> b(training_nodes.begin() + i, training_nodes.begin() + i + batch_size);
    cuslicer::device_vector<long> target(b);
    auto start = high_resolution_clock::now();
    ns->sample(target,(*s1));
    auto stop = high_resolution_clock::now();
    auto sampling_time = ((float)duration_cast<milliseconds>(stop - start).count())/1000;
    gpuErrchk(cudaDeviceSynchronize());

    EXPECT_LT(sampling_time, TARGET_TIME);
    // std::cout << "Sampling time " << duration <<"\n";
  }
  // s1->debug();
  cuslicer::transform::cleanup();
}


TEST(SAMPLING, correctness){
  cudaSetDevice(0);
// std::cout << "hello world\n";
  // std::string graph_name = "synth_8_2";
  std::string graph_name = "ogbn-arxiv";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false);
// std::cout << "Read synthetic dataset\n ";
// // // Test2: Construct simple k-hop neighbourhood sample.
// // // Sample datastructure.
  int num_layers = 1  ;
  Sample *s1  = new Sample(num_layers);
  vector<int> fanout({-1});
  bool self_edge = false;
  std::vector<long> training_nodes;
  for(int i=0;i<dataset->num_nodes;i++){
      training_nodes.push_back(i);
  }

  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, self_edge);
  cuslicer::device_vector<long> target(training_nodes);
  auto start = high_resolution_clock::now();
  ns->sample(target,(*s1));
  size_t edges = s1->block[1]->indices.size();
  EXPECT_EQ(edges , dataset->num_edges);
  auto stop = high_resolution_clock::now();
  auto sampling_time = ((float)duration_cast<milliseconds>(stop - start).count())/1000;
  gpuErrchk(cudaDeviceSynchronize());

    // EXPECT_LT(sampling_time, TARGET_TIME);
    // std::cout << "Sampling time " << duration <<"\n";
  // s1->debug();
  cuslicer::transform::cleanup();
}
