#include <iostream>
#include "dataset.h"
#include "models/model1.hh"
#include "samplers/full.h"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";

  auto *s = new TwoHopNoSampler(dataset->num_nodes, dataset->num_edges,
               dataset->indptr, dataset->indices, 1000,dataset->features, dataset->fsize);

  s->shuffle();

  auto* gcn1 = new SageAggr(dataset->fsize);
  auto* gcn2 = new SageAggr(dataset->fsize);
  // Test 1 print out all samples.
  int noB = s->number_of_batches();
  std::cout << "number of batches is" << noB <<"\n";

  for(int bid = 0; bid<noB;bid++){
    s->get_sample(bid);
    SampleLayer &l2 = s->sample.l2;
    Tensor<int> * indptr = new Tensor<int>(l2.indptr.data(),l2.indptr.size(),1);
    Tensor<int> * indices = new Tensor<int>(l2.indices.data(),l2.indices.size(),1);
    Tensor<float> * in = new Tensor<float>(s->sampled_features,l2.in_nodes,dataset->fsize);
    std::cout << "Current batch is " << bid <<"\n";

    auto t = gcn1->forwardPass(indptr, indices, (*in) , l2.out_nodes, l2.in_nodes);

    SampleLayer &l1 = s->sample.l1;
    Tensor<int> * indptr1 = new Tensor<int>(l1.indptr.data(),l1.indptr.size(),1);
    Tensor<int> * indices1 = new Tensor<int>(l1.indices.data(),l1.indices.size(),1);

    auto s1 = gcn2->forwardPass(indptr1, indices1, t , l1.out_nodes, l1.in_nodes);


    // gcn1->backwardPass(s);
    // delete indptr;
  }


  std::cout << "Forward and Backward working with memory management working !!\n";
  // Test 2 sample add one backward pass
  // Add loop code while also deleting tensors.
}
