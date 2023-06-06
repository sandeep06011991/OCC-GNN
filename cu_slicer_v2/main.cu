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
#include "graph/order_book.h"
using namespace std;
using namespace std::chrono;

int main(){

  cudaSetDevice(0);
  device_vector<PARTITIONIDX>::setLocalDevice(0);
  device_vector<NDTYPE>::setLocalDevice(0);
  
  // std::string graph_name = "synth_8_2";
  
  std::string graph_name = "ogbn-arxiv";

  std::string file = get_dataset_dir() + graph_name;
  
  int num_gpus = 4;
  bool random = false;
  std::shared_ptr<Dataset> dataset =\
      std::make_shared<Dataset>(file, num_gpus, random, true);

  std::shared_ptr<OrderBook> order =  std::make_shared<OrderBook>(get_dataset_dir(), graph_name, string("1GB"), 4);
  // Sample datastructure.
  int num_layers =3;
  Sample s1(num_layers);
  
  vector<int> fanout({20,20, 20});
  
  bool self_edge = false;
  std::vector<NDTYPE> training_nodes;
  for(int i=0;i<num_gpus * 1000 ;i++){
      training_nodes.push_back(i);
  }

  NeighbourSampler *ns  =  new NeighbourSampler(dataset, fanout, self_edge);

  cuslicer::device_vector<NDTYPE> target(training_nodes);
  ns->sample(target,s1);
  
  bool pull_optim = false;
  std::vector<NDTYPE> storage[8];
  int gpu_capacity[num_gpus];
  // Note how to handle this. ?
 for(int i=0;i < num_gpus; i++){
    gpu_capacity[i] = 0;
  // // Write a better version of this.
    for(int j = 0; j < num_gpus; j ++ ){
      for(int k = order->partition_offsets[j]; k < order->cached_offsets[i][j]; k ++ ){
        storage[i].push_back(k);
        gpu_capacity[i] ++;
      }
    }
  }
  
  //   // std::cout << "basic population done \n";
  //   // PushSlicer * sc2 = new PushSlicer(workload_map, storage, pull_optim, num_gpus, ns->dev_curand_states);
  //   // PartitionedSample ps1(num_layers, num_gpus);
  //   // sc1->slice_sample((*s1),ps1);
 
    PullSlicer * sc2 = new PullSlicer(order, num_gpus,\
         ns->dev_curand_states, dataset->num_nodes);
    PartitionedSample ps2(num_layers, num_gpus);
    bool loadbalancing = true;
    std::cout << "Sample \n";
    sc2->slice_sample(s1, ps2, loadbalancing);
    std::cout << "Slice \n";
    gpuErrchk(cudaDeviceSynchronize());
  // ps2.debug();
  // ps1.push_consistency();
    test_sample_partition_consistency(s1, ps2, storage, gpu_capacity, dataset->num_nodes, num_gpus);
  //   ps2.check_imbalance();
  //       loadbalancing = false;
  //       sc2->slice_sample(s1, ps2, loadbalancing);
  //       ps2.check_imbalance();
  // gpuErrchk(cudaDeviceSynchronize());
  
  // std::cout <<"All Done is consistent !\n";

  return 0;
}
