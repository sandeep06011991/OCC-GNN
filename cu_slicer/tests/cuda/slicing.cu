#include<iostream>
#include "../../graph/dataset.cuh"
#include "memory"
#include "../../graph/sample.h"
#include "../../samplers/samplers.h"
#include <vector>
#include "../../util/device_vector.h"
#include "../../util/cub.h"
#include "../../util/duplicate.h"
#include "../../transform/slice.h"
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;
#include "gtest/gtest.h"
#include <vector>
#include "../../util/environment.h"
#include "../../graph/sliced_sample.h"
#include "test.h"

#include <chrono>
#include <iostream>
using namespace cuslicer;

void test_sample_slice_consistency(int num_layers, std::vector<int> fanout, \
  int num_gpus,bool fully_cached){
  cudaSetDevice(0);
// std::cout << "hello world\n";
  // std::string graph_name = "synth_8_2";
  std::string graph_name = "ogbn-arxiv";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false, num_gpus);
// std::cout << "Read synthetic dataset\n ";
// // // Test2: Construct simple k-hop neighbourhood sample.
// // // Sample datastructure.
  Sample *s1  = new Sample(num_layers);
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
  cuslicer::transform::cleanup();
  cuslicer::device_vector<int> workload_map;
  std::vector<int> storage[8];
// // Test 3b. is_present = 1;
  int gpu_capacity[num_gpus];
  int batch_size = 4097;
  bool pull_optim = false;
  for(int i=0;i < num_gpus; i++)gpu_capacity[i] = 0;
// // Write a better version of this.
  workload_map = dataset->partition_map_d;
  for(int i=0;i<dataset->num_nodes;i++){
   #pragma unroll
    for(int j=0;j<num_gpus;j++){
      if(fully_cached){
        gpu_capacity[j]++;
           // in_f.push_back(nd%10);
        storage[j].push_back(i);
        // Since this case is all nodes are present
        // storage_map[j].push_back(i);
        }else{
        // storage_map[j].push_back(-1);
        }
      }
    }
    int rounds = 7;
    PushSlicer * sc1 = new PushSlicer(workload_map, storage, pull_optim, num_gpus);
    PartitionedSample ps1(num_layers, num_gpus);
    // s1->debug();
    for(int i=0; i < dataset->num_nodes - batch_size; i = i + batch_size){
      std::vector<long> b(training_nodes.begin() + i, training_nodes.begin() + i + batch_size);
      cuslicer::device_vector<long> target(b);
      auto start = high_resolution_clock::now();
        ns->sample(target,(*s1));
        sc1->slice_sample((*s1),ps1);
        test_sample_partition_consistency((*s1),ps1, storage, gpu_capacity, dataset->num_nodes, num_gpus);
        s1->clear();
        ps1.clear();
        // std::cout << "Sampling time " << duration <<"\n";
    }

}

TEST(SLICE_CORRECTNESS, LAYERS){
  int num_layers = 4;
  std::vector<int> fanout = {10,10,10,10};
  int num_gpus = 4;
  // test_sample_slice_consistency(num_layers, fanout,  num_gpus, true);
  test_sample_slice_consistency(num_layers, fanout,  num_gpus, false);

  cuslicer::transform::cleanup();
}

TEST(SLICE_CORRECTNESS, GPUS){
  int num_layers = 2;
  std::vector<int> fanout = {10,10};
  int num_gpus = 5;
  // test_sample_slice_consistency(num_layers, fanout,  num_gpus, true);
  test_sample_slice_consistency(num_layers, fanout,  num_gpus, false);
  cuslicer::transform::cleanup();
}
