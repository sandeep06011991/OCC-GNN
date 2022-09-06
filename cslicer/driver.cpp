#include <iostream>
#include <string>
#include "dataset.h"
#include "slicer.h"
#include <vector>
#include <cassert>
#include <chrono>
#include <iostream>

#include<thread>
#include<random>
#include<functional>
#include "util/environment.h"
using namespace std;
using namespace std::chrono;

// Keep this completely seperate.
int main(){
  std::string filename = "ogbn-arxiv";
  std::string DATA_DIR = get_dataset_dir();
  Dataset *dataset = new Dataset(DATA_DIR + filename);
  std::vector<int> *storage_map[4];
  long n = dataset->num_nodes;
  for(int i=0;i<4;i++){
    storage_map[i] = new std::vector<int>();
  }

  std::vector<int> workload_map;
  for(long j=0;j<dataset->num_nodes;j++){
    workload_map.push_back(j%4);
  }
  int gpu_capacity[] = {0,0,0,0};

  for(int i=0; i<4 ;i++){
      for(int j=0; j < n;j++){
        if(i==(j%4)){
          storage_map[i]->push_back(gpu_capacity[i]);
          gpu_capacity[i]++;
        }else{
          storage_map[i]->push_back(-1);
        }
      }
      assert(storage_map[i]->size() == n);
  }

  const int threads = 8;
  // const int threads = 1;
  Slicer *slicer = new Slicer(dataset, &workload_map, storage_map, gpu_capacity, 4096);
  vector<long int> vect{ 1000 };
  slicer->get_sample(vect);
  slicer->clear();

  std::cout <<"all minibatches processed.\n";
}
