#include<iostream>
#include "graph/dataset.cuh"
#include "util/environment.h"
#include "graph/sample.h"
#include "samplers/samplers.h"
#include <thrust/device_vector.h>
// #include "memory"
#include "transform/slice.h"
#include "graph/sliced_sample.h"
#include "tests/test.h"

int main(){

// Test1: Read graph datastructure.
cudaSetDevice(0);
std::string graph_name = "synth_8_2";
// std::string graph_name = "ogbn-arxiv";
std::string file = get_dataset_dir() + graph_name;
std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false);
std::cout << "Read synthetic dataset\n ";
// Test2: Construct simple k-hop neighbourhood sample.
// Sample datastructure.
  int num_layers = 1;
  Sample *s1  = new Sample(num_layers);
  bool pull_optim = false;
  int num_gpus = 2;
  vector<int> fanout({-1});
  bool self_edge = false;
  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, self_edge);
  thrust::host_vector<long> _training_nodes;
  for(int i=0;i<1;i++){
    _training_nodes.push_back(i);
  }
  thrust::device_vector<long> training_nodes;
  training_nodes = _training_nodes;
  ns->sample(training_nodes,(*s1));

// // assert(false);
// // sample_neighbourhood((*s), training_nodes, (*dataset));
// // Issues over memory who is responsibe for this.
// // Who creats it, uses it and destroys it.
// // What is its life size.
// // Test3 Create a work allocation
//
  thrust::device_vector<int> workload_map(dataset->num_nodes);
  thrust::device_vector<int> storage_map[4];
  thrust::device_vector<int> storage[4];
  int is_present =1;
// Test 3b. is_present = 1;
  int gpu_capacity[num_gpus];
  for(int i=0;i < num_gpus; i++)gpu_capacity[i] = 0;
// Write a better version of this.
  for(int i=0;i<dataset->num_nodes;i++){
  std::cout << dataset->partition_map[i] <<"\n";
   workload_map[i] = (dataset->partition_map[i]);
    #pragma unroll
    for(int j=0;j<num_gpus;j++){
      if(is_present == 1){
        gpu_capacity[j]++;
           // in_f.push_back(nd%10);
        storage[j].push_back(i);
        // Since this case is all nodes are present
        storage_map[j].push_back(i);
      }else{
        storage_map[j].push_back(-1);
      }
    }
  }
// //
// std::cout << "basic population done \n";
// int rounds = 4;
// PushSlicer *sc;
    PushSlicer * sc1 = new PushSlicer(workload_map, storage, pull_optim, num_gpus);
    PullSlicer * sc2 = new PullSlicer(workload_map, storage, pull_optim, num_gpus);
//   std::cout << "Slicer created \n";
//   s1->debug();
    PartitionedSample ps1(num_layers, num_gpus);
    PartitionedSample ps2(num_layers, num_gpus);

//   std::cout << "partition map created \n";
   sc1->slice_sample((*s1), ps1);
   // std::cout << "Push done \n";
   // ps1.debug();
   // sc1->slice_sample((*s1),ps2);

   ps2.debug();
   // std::cout << "Pull done \n";
   std::cout << "everything but cache managemnet done !\n";
// //   std::cout << "slicing done \n";
//   //ps.debug();
//
  test_sample_partition_consistency((*s1),ps2, storage, gpu_capacity, dataset->num_nodes, num_gpus);
  // test_pull_benefits(*s1, workload_map, storage, rounds);

  // test_reduction_communication_computation(*s1,workload_map,
  //           storage, storage_map,  rounds );
  // std::cout << "Hello World\n";
}
