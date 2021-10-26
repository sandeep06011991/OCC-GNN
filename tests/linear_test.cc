#include <iostream>
#include <string>
#include "layers/linear.hh"
#include "layers/relu.hh"
#include "tensor.hh"
#include <fstream>
#include <assert.h>

std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/linear";

int main(){

  int N = 19717;
  int M = 500;
  int K = 128;
  // N = 4;
  // M = 4;
  // K = 12;

  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (N * M  * sizeof(float));
  file1.read((char *)input , N * M  * sizeof(float));
  Tensor<float> * input_t = new Tensor<float>(input, N, M);

  std::fstream file2(BIN_DIR + "/W.bin",std::ios::in|std::ios::binary);
  float * W  = (float *)malloc (sizeof(float) * M * K);
  file2.read((char *)W, sizeof(float) * M * K);

  std::fstream file3(BIN_DIR + "/b.bin",std::ios::in|std::ios::binary);
  float * B  = (float *)malloc (sizeof(float) * K);
  file3.read((char *)B, sizeof(float) * K * 1);

  std::fstream file4(BIN_DIR + "/out2.bin",std::ios::in|std::ios::binary);
  float * output = (float *)malloc (N * K  * sizeof(float));
  file4.read((char *)output , N * K  * sizeof(float));
  Tensor<float> * output_t = new Tensor<float>(output, N, K);

  std::fstream file5(BIN_DIR + "/dW.bin",std::ios::in|std::ios::binary);
  float * dw  = (float *)malloc (sizeof(float) * M * K);
  file5.read((char *)dw, sizeof(float) * M * K);
  Tensor<float> * dw_t = new Tensor<float>(dw, M, K);

  std::fstream file6(BIN_DIR + "/db.bin",std::ios::in|std::ios::binary);
  float * db  = (float *)malloc (sizeof(float) * K);
  file6.read((char *)db, sizeof(float) * K * 1);
  Tensor<float> * db_t = new Tensor<float>(db, K, 1);

  std::fstream file7(BIN_DIR + "/dX.bin",std::ios::in|std::ios::binary);
  float * dX = (float *)malloc(sizeof(float) * N * M );
  file7.read((char *)dX , N * M  * sizeof(float));
  Tensor<float> * dX_t = new Tensor<float>(dX, N, M);


  LinearLayer* l1 = new LinearLayer(W,B, M, K);
  Relu* l2 = new Relu();
  // // Construct tensors
  // // Match forward pass
  // // check output at l2
  auto i = l1->forward(*input_t);
  auto out = l2->forward(i);
  auto g = l2->backward(*allocate_ones(N,K));
  auto dX_calc = l1->backward(g);
  //
  std::cout << "haha " << out.debugTensor() <<": " << output_t->debugTensor() <<"\n";
  assert(approx_equal(out.debugTensor(),output_t->debugTensor()));
  assert(approx_equal(dX_t->debugTensor(),dX_calc.debugTensor()));
  assert(approx_equal(dw_t->debugTensor(),l1->dW->debugTensor()));
  assert(approx_equal(db_t->debugTensor(),l1->db->debugTensor()));

  std::cout << "Calculated \n";
  l1->W->debugTensor();
  l1->W->viewTensor();
  // dX_t->debugTensor();
  std::cout << "Actual \n";
  // dX_calc.debugTensor();
  dw_t->debugTensor();
  dw_t->viewTensor();
  // l1->dW->debugTensor();
  // db_t->debugTensor();
  // l1->db->debugTensor();

    l1->update(.1);
}
