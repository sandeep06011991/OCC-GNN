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
#include "WorkerPool.h"
using namespace std;
using namespace std::chrono;

int main(){
  std::string filename = "ogbn-arxiv";
  std::string DATA_DIR = "/data/sandeep/";
  Dataset *dataset = new Dataset(DATA_DIR + filename);
  int num_epochs = 2;
  int minibatch_size = 1024;
  int num_workers = 10;
  // std::cout <<  "Hello world\n";
  WorkerPool *h = new WorkerPool(dataset->num_nodes, num_epochs,
      minibatch_size, num_workers);
  std::thread th(&WorkerPool::run, *h);
  int expected_samples = (dataset->num_nodes-1)/minibatch_size;
  for(int i=0;  i<expected_samples; i++){
      PySample * sh = h->generated_samples->pop_object();
      std::cout << "Read sample with " << sh->in_nodes << "\n";
  }
  th.join();
  std::cout << "Clean exit \n";
}

// Keep this completely seperate.
int sample_gen_main(){
  std::string filename = "ogbn-arxiv";
  std::string DATA_DIR = "/data/sandeep/";
  Dataset *dataset = new Dataset(DATA_DIR + filename);
  std::vector<int> storage_map[4];
  long n = dataset->num_nodes;

  std::vector<int> workload_map;
  for(long j=0;j<dataset->num_nodes;j++){
    workload_map.push_back(j%4);
  }
  for(int i=0; i<4 ;i++){
      for(int j=0; j < n;j++){
        if((j>=(n/4)*i) && ((j< (n/4)*(i+1)) || (i == 3))){
          storage_map[i].push_back(1);
        }else{
          storage_map[i].push_back(0);
        }
      }
      assert(storage_map[i].size() == n);
  }
  const int threads = 8;
  // const int threads = 1;
  Slicer *slicer[threads];
  for(int i=0;i<threads;i++){
    slicer[i] = new Slicer(dataset, workload_map, storage_map, 4096);
  }
  int b = slicer[0]->get_number_of_batches();
  // slicer->shuffle();

  std::cout << "Total minibatches " << b <<"\n";
  auto f = [](int tid, int b, int threads, Slicer **slicer){
    for(int i=tid;  i<  b  ;i=i + threads){
      // slicer[0]->get_sample(0);
      slicer[tid]->get_sample(i);
      // slicer[tid]->simple_3_hop_sample(i);
      slicer[tid]->clear();
      // slicer[0]->clear();
    }
    // std::cout << tid << "is done" <<"\n";
  };

  thread **th = (thread **)malloc(sizeof(thread) * threads);
  auto start =  high_resolution_clock::now();
  for(int i=0;i<threads;i++){
    th[i] = new thread(f,i, b, threads,slicer);
  }
  std::cout << "All threads started\n";
  for(int i=0;i<threads;i++){
    th[i]->join();
  }
  // for(int i=0;  i<b  ;i++){
  //   // std::cout << omp_get_thread_num() <<"\n";
  //   // slicer[0]->get_sample(0);
  //   slicer[0]->simple_3_hop_sample(i);
  //   slicer[0]->clear();
  // }
  auto stop = high_resolution_clock::now();
  auto duration = ((float)duration_cast<milliseconds>(stop - start).count())/1000;
  std::cout << "All sampling time " << duration <<"s\n";
  std::cout <<"all minibatches processed.\n";
}
