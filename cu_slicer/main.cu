#include<iostream>
#include "graph/dataset.cuh"
#include "util/environment.h"
#include "memory"
#include "graph/sample.h"
#include "samplers/samplers.h"

#include <vector>
#include "transform/slice.h"
#include "graph/sliced_sample.h"
#include "tests/cuda/test.h"
#include "util/device_vector.h"
#include "util/cub.h"
#include "util/duplicate.h"
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;
int main(){

// Test1: Read graph datastructure.

// test_duplicate();
  cudaSetDevice(0);
// std::cout << "hello world\n";
  // std::string graph_name = "synth_8_2";
  std::string graph_name = "ogbn-arxiv";
  std::string file = get_dataset_dir() + graph_name;
  int num_gpus = 6;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false, num_gpus);
// std::cout << "Read synthetic dataset\n ";
// // // Test2: Construct simple k-hop neighbourhood sample.
// // // Sample datastructure.

  int num_layers =4;
  Sample *s1  = new Sample(num_layers);
  vector<int> fanout({10,10,10,10});
  bool self_edge = false;
  for(int k = 0; k<10; k++){
  std::vector<long> training_nodes;
  for(int i=0;i<10000 ;i++){
      training_nodes.push_back(i);
  }

  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, self_edge);

  cuslicer::device_vector<long> target(training_nodes);
  ns->sample(target,(*s1));
  }
  bool pull_optim = false;

// //

  cuslicer::device_vector<int> workload_map;
  std::vector<int> storage[8];
  int is_present = 1 ;
// // Test 3b. is_present = 1;
  int gpu_capacity[num_gpus];
  workload_map = dataset->partition_map_d;
  for(int i=0;i < num_gpus; i++)gpu_capacity[i] = 0;
// // Write a better version of this.

  for(int i=0;i<dataset->num_nodes;i++){
   #pragma unroll
    for(int j=0;j<num_gpus;j++){
      if(is_present == 1){
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

// // std::cout << "basic population done \n";
    int rounds = 7;
    PushSlicer * sc1 = new PushSlicer(workload_map, storage, pull_optim, num_gpus);
    PartitionedSample ps1(num_layers, num_gpus);
    //
    // std::cout <<"Reached erere1\n";
    std::cout << "Start slicing\n";
    sc1->slice_sample((*s1),ps1);


    // sc1->slice_sample((*s1),ps1);
    // std::cout <<"Reached erere3\n";

//     // PullSlicer * sc2 = new PullSlicer(workload_map, storage, pull_optim, num_gpus);
//    std::cout << "Slicer created \n";
//
//     PartitionedSample ps2(num_layers, num_gpus);
//     // s1->debug();
//
// //   std::cout << "partition map created \n";
//    sc1->slice_sample((*s1), ps2);
//      // ps2.debug();
//    // std::cout << "Push done \n";
   // ps1.debug();
   // return 0;
//    //
//
//    // ps2.debug();
//    // std::cout << "Pull done \n";
//    // std::cout << "everything but cache managemnet done !\n";
// // //   std::cout << "slicing done \n";
  // ps1.debug();
    ps1.push_consistency();
// //
    test_sample_partition_consistency((*s1),ps1, storage, gpu_capacity, dataset->num_nodes, num_gpus);
  //

  cuslicer::transform::cleanup();
  std::cout <<"All Done is consistent !\n";

  return 0;
//   // test_pull_benefits(*s1, workload_map, storage, rounds);
//
//   // test_reduction_communication_computation(*s1,workload_map,
//   //           storage, storage_map,  rounds );
  // std::cout << "Hello World\n";
}
