#include<iostream>
#include "graph/dataset.h"
#include "util/environment.h"
#include "graph/sample.h"
#include "samplers/samplers.h"
#include <thrust/device_vector.h>
// #include "memory"
#include "transform/slice.h"
#include "graph/sliced_sample.h"
//#include "tests/test.h"

int main(){

// Test1: Read graph datastructure.
cudaSetDevice(0);
std::string graph_name = "ogbn-arxiv";
std::string file = get_dataset_dir() + graph_name;
std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false);

// Test2: Construct simple k-hop neighbourhood sample.
// Sample datastructure.
int num_layers = 3;
Sample *s1  = new Sample(num_layers);

thrust::host_vector<long> _training_nodes;
for(int i=0;i<3;i++){
  _training_nodes.push_back(i);
}
thrust::device_vector<long> training_nodes;
training_nodes = _training_nodes;
thrust::device_vector<long> v1;
v1.resize(0);
std::cout << "Verify" << thrust::raw_pointer_cast(v1.data()) <<"\n";
int fanout = 3;
bool self_edge = true;
NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, false, self_edge);
std::cout << "starting to sample\n";
ns->sample(training_nodes,(*s1));
// s1->debug();
// assert(false);
// sample_neighbourhood((*s), training_nodes, (*dataset));
// Issues over memory who is responsibe for this.
// Who creats it, uses it and destroys it.
// What is its life size.
// Test3 Create a work allocation

thrust::device_vector<int> workload_map(dataset->partition_map
    ,dataset->partition_map + dataset->num_nodes);;
thrust::device_vector<int> storage_map[4];
thrust::device_vector<int> storage[4];
int is_present =0;
// Test 3b. is_present = 1;
int gpu_capacity[4];
for(int i=0;i < 4; i++)gpu_capacity[i] = 0;
// Write a better version of this.
for(int i=0;i<dataset->num_nodes;i++){
//   std::cout << dataset->partition_map[i] <<"\n";
//   workload_map.insert(dataset->partition_map[i]);
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
//
  std::cout << "basic population done \n";
  int rounds = 4;
  bool pull_optim = false;
  int num_gpus = 4;
  std::cout << "Create zlicer\n";
  Slice * sc = new Slice(workload_map, storage, true, rounds, pull_optim, num_gpus);
  std::cout << "Slicer created \n";
//   // s1->debug();
  PartitionedSample ps(num_layers);
  std::cout << "partition map created \n";
  sc->slice_sample((*s1), ps);
//   std::cout << "slicing done \n";
  //ps.debug();

  // test_sample_partition_consistency_gat((*s1),ps, storage, gpu_capacity, dataset->num_nodes);
  // test_pull_benefits(*s1, workload_map, storage, rounds);

  // test_reduction_communication_computation(*s1,workload_map,
  //           storage, storage_map,  rounds );
  // std::cout << "Hello World\n";
  }
