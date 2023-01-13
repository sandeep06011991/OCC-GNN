#include<iostream>
#include "graph/dataset.h"
#include "util/environment.h"
#include "graph/sample.h"
#include "samplers/samplers.h"
// #include "memory"
#include "transform/slice.h"
#include "tests/test.h"

int main(){

  // Test1: Read graph datastructure.
  std::string graph_name = "ogbn-arxiv";
  std::string file = get_dataset_dir() + graph_name;
  int num_partitions = -1;
  
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false, num_partitions);
  if (num_partitions == -1){
  	num_partitions = 4;
  }
  // Test2: Construct simple k-hop neighbourhood sample.
  // Sample datastructure.
  int num_layers = 4;
  Sample *s1  = new Sample(num_layers);
  s1->debug();
  std::vector<long> training_nodes;
  for(int i=0;i<10;i++){
    training_nodes.push_back(i);
  }
  std::cout << training_nodes.size() <<"\n";
  int fanout = 3;
  bool self_edge = false;
  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, false, self_edge);
  ns->sample(training_nodes,(*s1));

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
  bool pull_optim = false;
  Slice * sc = new Slice(workload_map, storage, self_edge, rounds, pull_optim, num_partitions);
  //s1->debug();
  PartitionedSample ps(num_layers, num_partitions);
  sc->slice_sample((*s1), ps);
  std::cout << "slicing done \n";
  //ps.debug();

  test_sample_partition_consistency((*s1),ps, storage, gpu_capacity, dataset->num_nodes, num_partitions);
  // test_pull_benefits(*s1, workload_map, storage, rounds);

  // test_reduction_communication_computation(*s1,workload_map,
  //           storage, storage_map,  rounds );
  // std::cout << "Hello World\n";
}
