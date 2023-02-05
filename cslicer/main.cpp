#include<iostream>
#include "graph/dataset.h"
#include "util/environment.h"
#include "graph/sample.h"
#include "samplers/samplers.h"
// #include "memory"
#include "transform/slice.h"
#include "tests/test.h"
#include <algorithm>
#include <ctime>
#include <chrono>
using namespace std::chrono;

int main(){

  // Test1: Read graph datastructure.
  std::string graph_name = "ogbn-products";
  std::string file = get_dataset_dir() + graph_name;
  int num_partitions = 4;

  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false, num_partitions);
  if (num_partitions == -1){
  	num_partitions = 4;
  }
  // Test2: Construct simple k-hop neighbourhood sample.
  // Sample datastructure.
  int num_layers = 3;
  Sample *s1  = new Sample(num_layers);
  s1->debug();
  std::vector<long> training_nodes;

  for(int i=0;i<dataset->num_nodes;i++){
    training_nodes.push_back(i);
  }
  
  std::cout << training_nodes.size() <<"\n";
  int fanout = 10;
  // For GAT self_edge = True, GCN No self edge
  // bool self_edge = true;
  // bool pull_optim = false;
  // For GCN
  bool self_edge = false;
  bool pull_optim = true;

  bool deterministic = false;
  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, deterministic, self_edge);
  std::vector<long> sample;
  for(int k=0;k<4;k++){
  random_shuffle(training_nodes.begin(), training_nodes.end());
  auto start2 = high_resolution_clock::now();

  for(long i=0;i<dataset->num_nodes;i=i+1032){
  	long j = std::min(i + 1032, dataset->num_nodes - 1);
	copy(training_nodes.begin()+i, training_nodes.begin() + j, back_inserter(sample));
  	ns->sample(sample,(*s1));
	sample.clear();
	std::cout << i <<":"<<dataset->num_nodes <<"\n";
  }
   auto start3 = high_resolution_clock::now();
   auto duration1 = duration_cast<milliseconds>(start3- start2);
  std::cout << "Sample time" << duration1.count() <<"\n";
  }
  return 0;
    // sample_neighbourhood((*s), training_nodes, (*dataset));
  // Issues over memory who is responsibe for this.
  // Who creats it, uses it and destroys it.
  // What is its life size.

  // Test3 Create a work allocation
  std::vector<int> workload_map;
  std::vector<int> storage_map[num_partitions];
  std::vector<int> storage[num_partitions];
  // Test 3a.
  int is_present =0;
  // Test 3b. is_present = 1;
  int gpu_capacity[num_partitions];
  for(int i=0;i < num_partitions; i++)gpu_capacity[i] = 0;
  for(int i=0;i<dataset->num_nodes;i++){
    workload_map.push_back(dataset->partition_map[i]);
#pragma unroll
    for(int j=0;j<num_partitions;j++){
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

  // for(int j=0;j<4;j++){
  //   storage[j].push_back(100);
  //   storage_map[j][100] = 0;
  //   gpu_capacity[j]++;
  // }
  int rounds = 4;

  Slice * sc = new Slice(workload_map, storage, self_edge, rounds, pull_optim, num_partitions);
  s1->debug();
  PartitionedSample ps(num_layers, num_partitions);
  sc->slice_sample((*s1), ps);
  std::cout << "slicing done \n";
  //ps.debug();

  test_sample_partition_consistency((*s1),ps, storage, gpu_capacity, dataset->num_nodes, num_partitions);
  // test_sample_partition_consistency_gat((*s1),ps, storage, gpu_capacity, dataset->num_nodes, num_partitions);
  // test_pull_benefits(*s1, workload_map, storage, rounds);

  // test_reduction_communication_computation(*s1,workload_map,
  //           storage, storage_map,  rounds );
  // std::cout << "Hello World\n";
}
