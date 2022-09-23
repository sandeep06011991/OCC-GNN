#include<iostream>
#include "graph/dataset.h"
#include "util/environment.h"
#include "graph/sample.h"
#include "samplers/samplers.h"
#include "memory"
#include "transform/slice.h"
#include "tests/test.h"
int main(){

  // Test1: Read graph datastructure.
  std::string graph_name = "ogbn-arxiv";
  std::string file = get_dataset_dir() + graph_name;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file);

  // Test2: Construct simple k-hop neighbourhood sample.
  // Sample datastructure.
  Sample *s  = new Sample(3);
  std::vector<long> training_nodes{0,1,2,3};
  int fanout = 2;
  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout);
  ns->sample(training_nodes,(*s));
  // sample_neighbourhood((*s), training_nodes, (*dataset));
  // Issues over memory who is responsibe for this.
  // Who creats it, uses it and destroys it.
  // What is its life size.

  // Test3 Create a work allocation
  std::vector<int> workload_map;
  std::vector<int> storage_map[4];
  std::vector<int> storage[4];
  // Test 3a.
  int is_present = 1;
  // Test 3b. is_present = 1;
  int gpu_capacity[4];
  for(int i=0;i < 4; i++)gpu_capacity[i] = 0;
  for(int i=0;i<dataset->num_nodes;i++){
    workload_map.push_back(i%4);

    #pragma unroll
    for(int j=0;j<4;j++){
      if(is_present == 1){
        gpu_capacity[j]++;
        storage[j].push_back(i);
        // Since this case is all nodes are present
        storage_map[j].push_back(i);
      }else{
        storage_map[j].push_back(-1);
      }
    }
  }


  Slice * sc = new Slice(workload_map, storage_map);
  PartitionedSample ps;
  sc->slice_sample((*s), ps);
  test_sample_partition_consistency((*s),ps, storage, gpu_capacity, dataset->num_nodes);
  std::cout << "Hello World\n";
}
