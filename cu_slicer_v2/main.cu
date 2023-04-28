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
#include "util/types.h"
using namespace std;
using namespace std::chrono;

int main(){

  cudaSetDevice(0);
  std::string graph_name = "synth_8_2";
  
  // std::string graph_name = "ogbn-arxiv";

  std::string file = get_dataset_dir() + graph_name;
  
  int num_gpus = 2;
  bool random = false;
  std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(file, false, num_gpus, random);

  // Sample datastructure.
  int num_layers =2;
  Sample *s1  = new Sample(num_layers);
  vector<int> fanout({20, 20});
  
  bool self_edge = false;
  std::vector<NDTYPE> training_nodes;
  for(int i=0;i<num_gpus ;i++){
      training_nodes.push_back(i);
  }

  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, self_edge);

  cuslicer::device_vector<NDTYPE> target(training_nodes);
  ns->sample(target,(*s1));
  s1->debug();
  bool pull_optim = false;

  cuslicer::device_vector<PARTITIONIDX> workload_map;
  std::vector<NDTYPE> storage[8];
  int is_present = 1 ;
// // Test 3b. is_present = 1;
  int gpu_capacity[num_gpus];
  workload_map = dataset->partition_map_d;
  workload_map.debug("workload map");
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

    // std::cout << "basic population done \n";
    // PushSlicer * sc1 = new PushSlicer(workload_map, storage, pull_optim, num_gpus);
    // PartitionedSample ps1(num_layers, num_gpus);
    // sc1->slice_sample((*s1),ps1);
 
    PullSlicer * sc2 = new PullSlicer(workload_map, storage, pull_optim, num_gpus,\
         ns->dev_curand_states);
      PartitionedSample ps2(num_layers, num_gpus);
       sc2->slice_sample((*s1), ps2);
    // ps2.debug();
    // ps1.push_consistency();
    test_sample_partition_consistency((*s1),ps2, storage, gpu_capacity, dataset->num_nodes, num_gpus);
  gpuErrchk(cudaDeviceSynchronize());
  cuslicer::transform<NDTYPE>::cleanup();
  cuslicer::transform<PARTITIONIDX>::cleanup();
  
  std::cout <<"All Done is consistent !\n";

  return 0;
}
