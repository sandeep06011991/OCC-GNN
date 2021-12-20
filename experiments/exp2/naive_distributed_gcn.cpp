#include<iostream>
#include "util/gpu.hh"
#include "util/tensor.hh"
#include "gnn/sage.hh"
#include "samplers/sample.h"
#include "samplers/neigh_sample.h"
#include "util/timer.h"
#include "dataset.h"
int no_gpus = 4;

int run_experiment(string filename){
  // Dummy dataset
  // std::string filename = "pubmed";
  std::string BIN_DIR = "/home/spolisetty/data/";
  reset_timers();
  active_timer(MOVEMENT_COST);
  active_timer(MOVEMENT_COMPUTE);
  // Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  Dataset * dataset = new Dataset(BIN_DIR + filename);
  int no_vertices = dataset->num_nodes;
  int no_edges = dataset->num_edges;
  int fsize = dataset->fsize;

//   int no_gpus = 4;
//   enable_peer_communication();
  auto *s = new TwoHopNeighSampler(dataset->num_nodes, dataset->num_edges,
             dataset->indptr, dataset->indices, 1024,dataset->features,
              dataset->labels, dataset->fsize);
  SageAggr * saggr_layer1[4];
  SageAggr * saggr_layer2[4];
  for(int device = 0;device < 4; device ++) {
    saggr_layer1[device] = new SageAggr(dataset->fsize, device);
    saggr_layer2[device] = new SageAggr(dataset->fsize, device);
  }
  int noB = s->number_of_batches();
  // noB = 10;
  for(int bid = 0; bid<noB/4;bid++){
    std::cout <<  "batch " << bid <<"\n";
    size_t free, total;
    cudaMemGetInfo( &free, &total );
    // std::cout << "GPU  memory: free=" << free << ", total=" << total << std::endl;
    for(int device = 0;device < 4; device ++) {
      if(bid*4 + device > noB)continue;
      s->get_sample(bid * 4 + device);
      // Create tensor data from samples.
      SampleLayer &l2 = s->sample.l2;
      Tensor<int> * indptr1 = new Tensor<int>(l2.indptr.data(),Shape(l2.indptr.size(),1),device);
      Tensor<int> * indices1 = new Tensor<int>(l2.indices.data(),Shape(l2.indices.size(),1),device);
      start_timer(MOVEMENT_COST);
      Tensor<float> * batch_in = new Tensor<float>(s->batch_features,Shape(l2.in_nodes,dataset->fsize),device);
      stop_timer(MOVEMENT_COST);
      SampleLayer &l1 = s->sample.l1;
      Tensor<int> * indptr2 = new Tensor<int>(l1.indptr.data(),Shape(l1.indptr.size(),1), device);
      Tensor<int> * indices2 = new Tensor<int>(l1.indices.data(), Shape(l1.indices.size(),1), device);
      start_timer(MOVEMENT_COST);
      // Tensor<int> * batch_labels = new Tensor<int>(s->batch_labels, Shape(l1.out_nodes, 1);
      stop_timer(MOVEMENT_COST);
      start_timer(MOVEMENT_COMPUTE);
      assert(l1.in_nodes == l2.out_nodes);
      auto out = saggr_layer1[device]->forward(*indptr1,*indices1, *batch_in, l2.out_nodes,l2.in_nodes);
      saggr_layer2[device]->forward(*indptr2,*indices2, *out ,l1.out_nodes, l1.in_nodes);
      stop_timer(MOVEMENT_COMPUTE);
      indptr1->clearTensor();
      indptr2->clearTensor();
      indices1->clearTensor();
      indices2->clearTensor();
      batch_in->clearTensor();
    }
     sync_all_gpus();
  }
  print_timer();
}

int main(int argc, char *argv[]){
  std::string filename = (argv[1]);
  run_experiment(filename);
}
