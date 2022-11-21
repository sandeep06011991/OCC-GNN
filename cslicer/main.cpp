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
  std::string graph_name = "ogbn-products";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false);

  // Test2: Construct simple k-hop neighbourhood sample.
  // Sample datastructure.
  int num_layers = 3;
  Sample *s1  = new Sample(num_layers);

  std::vector<long> training_nodes;
  for(int i=0;i<4096;i++){
    training_nodes.push_back(i);
  }
  std::cout << training_nodes.size() <<"\n";
  int fanout = 10;
  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, false);
  ns->sample(training_nodes,(*s1));

  // sample_neighbourhood((*s), training_nodes, (*dataset));
  // Issues over memory who is responsibe for this.
  // Who creats it, uses it and destroys it.
  // What is its life size.

  // Test3 Create a work allocation
  std::vector<int> workload_map;
  std::vector<int> storage_map[4];
  std::vector<int> storage[4];
  // Test 3a.
  int is_present =0;
  // Test 3b. is_present = 1;
  int gpu_capacity[4];
  for(int i=0;i < 4; i++)gpu_capacity[i] = 0;
  for(int i=0;i<dataset->num_nodes;i++){
    workload_map.push_back(dataset->partition_map[i]);
    #pragma unroll
    for(int j=0;j<4;j++){
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
  bool pull_optim = true;
  // Slice * sc = new Slice(workload_map, storage, false, rounds, pull_optim);
  // // s->debug();
  // PartitionedSample ps(num_layers);
  // sc->slice_sample((*s1), ps);
  // std::cout << "slicing done \n";
  // ps.debug();
  // test_sample_partition_consistency((*s1),ps, storage, gpu_capacity, dataset->num_nodes);
  // test_pull_benefits(*s1, workload_map, storage, rounds);

  test_reduction_communication_computation(*s1,workload_map,
            storage, storage_map,  rounds );
  // std::cout << "Hello World\n";
}
