#include <iostream>
#include "dataset.h"
#include "models/model1.hh"
#include "layers/relu.hh"
#include "samplers/full.h"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";
  // Read all test input data.
  int M = dataset->fsize;
  int N = 128;
  int K = dataset->noClasses;
  std::string TARGET_DIR = "/mnt/homes/spolisetty/data/tests/gcn";

  std::fstream file2(TARGET_DIR + "/W1.bin",std::ios::in|std::ios::binary);
  float * W1  = (float *)malloc (sizeof(float) * M * N);
  file2.read((char *)W1, sizeof(float) * M * N);
  Tensor<float> * W_ten = new Tensor<float>(W1,M,N);
  // W_ten->viewTensor();
  // assert(false);

  std::fstream file3(TARGET_DIR + "/b1.bin",std::ios::in|std::ios::binary);
  float * B1  = (float *)malloc (sizeof(float) * N);
  file3.read((char *)B1, sizeof(float) * N * 1);

  std::fstream file4(TARGET_DIR + "/W2.bin",std::ios::in|std::ios::binary);
  float * W2  = (float *)malloc (sizeof(float) * N * K);
  file4.read((char *)W2, sizeof(float) * N * K);

  std::fstream file5(TARGET_DIR + "/b2.bin",std::ios::in|std::ios::binary);
  float * B2  = (float *)malloc (sizeof(float) * K);
  file5.read((char *)B2, sizeof(float) * K * 1);

  std::fstream file6(TARGET_DIR + "/predicted.bin",std::ios::in|std::ios::binary);

  float * pred  = (float *)malloc (sizeof(float) * dataset->num_nodes * dataset->noClasses);
  file6.read((char *)pred, sizeof(float) * dataset->num_nodes * dataset->noClasses);

  Tensor<float> * correct_out = new Tensor<float>(pred ,dataset->num_nodes,dataset->noClasses);

  auto  model = new GCNBasic(dataset->fsize, N, dataset->noClasses, W1, B1,W2,B2);
  Tensor<int> * indptr1 = new Tensor<int>(dataset->indptr,dataset->num_nodes,1);
  Tensor<int> * indices1 = new Tensor<int>(dataset->indices,dataset->num_edges ,1);
  Tensor<float> * batch_in = new Tensor<float>(dataset->features,dataset->num_nodes,dataset->fsize);
  Tensor<int> * indptr2 = new Tensor<int>(dataset->indptr,dataset->num_nodes,1);
  Tensor<int> * indices2 = new Tensor<int>(dataset->indices,dataset->num_edges ,1);
  Tensor<int> * batch_labels = new Tensor<int>(dataset->labels,dataset->num_nodes,1);
  //
  Tensor<float>& out = model->forward(*indptr1, *indices1, *indptr2, *indices2, \
        *batch_in , *batch_labels);


  std::cout << "Total forward sum" << out.debugTensor();
  std::cout << "Total actual forward sum" << correct_out-> debugTensor();

  auto loss = new CrossEntropyLoss();
  auto total = loss->forward(out,*batch_labels);

  std::cout << "Total bce loss forward"<< total.debugTensor() <<"\n";
  Tensor<float>& grad = loss->backward(*batch_labels);
  grad.debugTensor();
  grad.viewTensor();
  model->backward(grad);
  std::cout << "bce out"<< grad.debugTensor()<<"\n";
  std::cout << "W1  " << model->ll1->dW->debugTensor() <<"\n";
  std::cout << "b1  " << model->ll1->db->debugTensor() <<"\n";
  std::cout << "W2  " << model->ll2->dW->debugTensor() <<"\n";
  std::cout << "b2  " << model->ll2->db->debugTensor() <<"\n";
  std::cout << "Total Loss is sum" << total.debugTensor()/dataset->num_nodes <<"\n";

  // assert(approx_equal(loss.debugTensor(),correct_out->debugTensor()));
    // std::cout << "total sum " << correct_out->debugTensor();
  // std::cout << "backward OK!\n";
    // indptr1->cleanUpTensor();
    // indices1->cleanUpTensor();
    // batch_in->cleanUpTensor();
    // indptr2->cleanUpTensor();
    // indices2->cleanUpTensor();
    // batch_labels->cleanUpTensor();
    // delete indptr1, indices1, batch_in, indptr2, indices2, batch_labels;

  // Test 2 sample add one backward pass
  // Add loop code while also deleting tensors.
}
