#include <iostream>
#include <string>
#include "dataset.h"
#include <fstream>
#include "tensor.hh"
// #include "samplers/full.h"
#include "gnn/sage.hh"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";

  std::fstream file1(BIN_DIR + "/aggr.bin",std::ios::in|std::ios::binary);
  float * aggr = (float *)malloc (dataset->num_nodes * dataset->fsize * sizeof(float));
  file1.read((char *)aggr , dataset->num_nodes * dataset->fsize * sizeof(float));
  Tensor<float> * aggr_t = new Tensor<float>(aggr, dataset->num_nodes, dataset->fsize);

  std::fstream file2(BIN_DIR + "/grad.bin",std::ios::in|std::ios::binary);
  float * grad  = (float *)malloc (sizeof(float) * dataset->num_nodes * dataset->fsize);
  file2.read((char *)grad, dataset->num_nodes * dataset->fsize * sizeof(float));
  Tensor<float> * grad_t = new Tensor<float>(grad, dataset->num_nodes, dataset->fsize);

  Tensor<int> * offsets = new Tensor<int>(dataset->indptr,dataset->num_nodes+1,1);
  Tensor<int> * indices = new Tensor<int>(dataset->indices,dataset->num_edges,1);
  Tensor<float> * in_data = new Tensor<float>(dataset->features,dataset->num_nodes, \
      dataset->fsize);

  SageAggr * l1 =new SageAggr(dataset->fsize);
  auto out = l1->forwardPass(offsets, indices, *in_data, dataset->num_nodes, dataset->num_nodes);
  auto grad_in =  allocate_ones(dataset->num_nodes, dataset->fsize);
  auto grad_out = l1->backwardPass(grad_in);
  out.debugTensor();
  aggr_t->debugTensor();
  grad_out->debugTensor();
  grad_t->debugTensor();

  // auto sampler = new TwoHopNoSample(dataset->num_nodes, dataset->num_edges,
  //   dataset->indptr, dataset->indices, 100);
  // sampler->shuffle();

  std::cout << "Hello World ! \n";
}
