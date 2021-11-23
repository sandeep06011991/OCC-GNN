#include <iostream>
#include "dataset.h"
#include "models/model1.hh"
#include "models/model2.hh"
#include "layers/relu.hh"
#include "samplers/full.h"
#include <assert.h>
#include "layers/accuracy.hh"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";
  std::cout << " classes: " << dataset->noClasses << "\n";
  // auto *s = new TwoHopNoSampler(dataset->num_nodes, dataset->num_edges,
  //              dataset->indptr, dataset->indices, 6,dataset->features,
  //               dataset->labels, dataset->fsize);

  // auto  model = new GCNBasic(dataset->fsize, 256, dataset->noClasses);
  auto model = new TwoLayerModel(dataset->fsize, 256, dataset->noClasses);
  // auto* gcn1 = new SageAggr(dataset->fsize);
  // auto* gcn2 = new SageAggr(dataset->fsize);
  // // Test 1 print out all samples.
  // std::cout << "number of batches is" << noB <<"\n";
  // int noB = s->number_of_batches();
  // int noB = 1;
  int batch_size = 128;
// Perfect debugging test.
  int indptr[] = {};
  int indices[] = {};
  CrossEntropyLoss* loss = new CrossEntropyLoss();
  Tensor<float> * batch_in = new Tensor<float>(dataset->features, batch_size,dataset->fsize);
  Tensor<int> * labels = new Tensor<int>(dataset->labels, batch_size, 1);

  for(int i=0;i<100;i++){
  for(int bid = 0; bid<1;bid++){
    // s->get_sample(bid);
    size_t free, total;
    cudaMemGetInfo( &free, &total );
    std::cout << "GPU  memory: free=" << free << ", total=" << total << std::endl;
    std::cout << "Sample : " << i<<"\n";
      // SampleLayer &l2 = s->sample.l2;
    // Tensor<int> * indptr1 = new Tensor<int>(l2.indptr.data(),l2.indptr.size(),1);
    // Tensor<int> * indices1 = new Tensor<int>(l2.indices.data(),l2.indices.size(),1);
    // Tensor<float> * batch_in = new Tensor<float>(s->batch_features,l2.in_nodes,dataset->fsize);
    // SampleLayer &l1 = s->sample.l1;
    // Tensor<int> * indptr2 = new Tensor<int>(l1.indptr.data(),l1.indptr.size(),1);
    // Tensor<int> * indices2 = new Tensor<int>(l1.indices.data(),l1.indices.size(),1);
    // Tensor<int> * batch_labels = new Tensor<int>(s->batch_labels, l1.out_nodes, 1);
    // std::cout << "Full batch sampling \n";
    // Tensor<int> * indptr1 = new Tensor<int>(dataset->indptr,dataset->num_nodes,1);
    // Tensor<int> * indices1 = new Tensor<int>(dataset->indices,dataset->num_edges ,1);
    // Tensor<float> * batch_in = new Tensor<float>(dataset->features,dataset->num_nodes,dataset->fsize);
    // Tensor<int> * indptr2 = new Tensor<int>(dataset->indptr,dataset->num_nodes,1);
    // Tensor<int> * indices2 = new Tensor<int>(dataset->indices,dataset->num_edges ,1);
    // Tensor<int> * batch_labels = new Tensor<int>(dataset->labels,dataset->num_nodes,1);

    // Tensor<float>& out = model->forward(*indptr1, *indices1, *indptr2, *indices2, \
    //       *batch_in , *batch_labels);
    Tensor<float>& out = model->forward( *batch_in );
    std::cout << "Calculate loss \n";
    Tensor<float>& total_loss = loss->forward(out, *labels);
    Tensor<float> & grad_out = loss->backward(*labels);
    model->backward(grad_out);
      std::cout << "Calculate loss \n";
    float loss = total_loss.debugTensor();
    std::cout << "printing loss " << total_loss.debugTensor()/total_loss.dim1 <<" \n ";
    std::cout << "accuracy is (expect close to one):" <<  find_accuracy(*labels, out) <<" xx \n";
    // if(loss != loss){
    //   assert(false);
    // }
    // delete batch_in;
    // model->backward(out);

    model->update(.01);
    // std::cout << "backward OK!\n";
    // indptr1->clearTensor();
    // indices1->clearTensor();
    // batch_in->clearTensor();
    // indptr2->clearTensor();
    // indices2->clearTensor();
    // batch_labels->clearTensor();
    // delete indptr1, indices1, batch_in, indptr2, indices2, batch_labels;

  // }
  }
}

  std::cout << "Forward and Backward working with memory management working !!\n";
  // Test 2 sample add one backward pass
  // Add loop code while also deleting tensors.
}
