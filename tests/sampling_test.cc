#include <iostream>
#include <string>
#include "dataset.h"
#include <fstream>
#include "tensor.hh"
#include "samplers/full.h"
#include "gnn/sage.hh"
// #include "gnn/sage.hh"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";
  std::cout << " classes: " << dataset->noClasses << "\n";
  int batch_size = 4;
  auto * s = new TwoHopNoSampler(dataset->num_nodes, dataset->num_edges,
             dataset->indptr, dataset->indices, batch_size,dataset->features,
              dataset->labels, dataset->fsize);

  float * aggr2 = (float *)malloc(sizeof(float) * batch_size * dataset->fsize);
  memset(aggr2, 0, sizeof(float) * batch_size * dataset->fsize);
  std::cout << "printing neighbourhood \n";
  int batch_id = 2;
  for(int k=0;k<batch_size;k++){
    int i = batch_id * batch_size + k;
    std::cout << i <<":";
    for(int nb_offset = dataset->indptr[i]; nb_offset < dataset->indptr[i+1];nb_offset++){
      int nb1_id = dataset->indices[nb_offset];
      std::cout <<nb1_id <<"(";
      for(int nb_offsets1 = dataset->indptr[nb1_id]; nb_offsets1 < dataset->indptr[nb1_id +1];
          nb_offsets1++){
        int nb2_id = dataset->indices[nb_offsets1];
        for(int f = 0;f<dataset->fsize;f++){
          aggr2[k*dataset->fsize+f] += dataset->features[nb2_id*dataset->fsize+f];
        }
        std::cout << nb2_id <<" ";
      }
      std::cout << ")";
    }
    std::cout <<"\n";
  }
  Tensor<float> * aggr_t = new Tensor<float>(aggr2, batch_size, dataset->fsize);
  SageAggr * layer1 = new SageAggr(dataset->fsize);
  SageAggr * layer2 = new SageAggr(dataset->fsize);

  // s->shuffle();
  s->get_sample(2);
  SampleLayer &l2 = s->sample.l2;
  Tensor<int> * indptr1 = new Tensor<int>(l2.indptr.data(),l2.indptr.size(),1);
  Tensor<int> * indices1 = new Tensor<int>(l2.indices.data(),l2.indices.size(),1);
  Tensor<float> * batch_in = new Tensor<float>(s->batch_features,l2.in_nodes,dataset->fsize);
  SampleLayer &l1 = s->sample.l1;
  Tensor<int> * indptr2 = new Tensor<int>(l1.indptr.data(),l1.indptr.size(),1);
  Tensor<int> * indices2 = new Tensor<int>(l1.indices.data(),l1.indices.size(),1);
  Tensor<int> * batch_labels = new Tensor<int>(s->batch_labels, l1.out_nodes, 1);

  // ensor<float>& SageAggr::forward(Tensor<int>& offsets , Tensor<int>& indices,
  //         Tensor<float>& in, int num_nodes_out, int num_nodes_in){
  auto o1 = layer1->forward(*indptr1, *indices1, *batch_in, indptr1->dim1-1,batch_in->dim1);
  auto o2 = layer2->forward(*indptr2,* indices2, o1, indptr2->dim1-1,o1.dim1);
  std::cout << "total sum" << o2.debugTensor() <<"\n";
  o2.viewTensor();
  std::cout << "total sum" << aggr_t->debugTensor() <<"\n";
  aggr_t->viewTensor();
  //   // Test 1 print out all samples.
  // int noB = s->number_of_batches();
  // // std::cout << "number of batches is" << noB <<"\n";
  // // for(int bid = 0; bid<noB;bid++){
  //   s->get_sample(0);
  // }


  // Test 2 print out all csr neighbourhoods.


  std::cout << "Hello World ! \n";
}
