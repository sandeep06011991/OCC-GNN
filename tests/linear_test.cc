#include <iostream>
#include <string>
#include "layers/loss.hh"
#include "tensor.hh"
#include <fstream>

std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/linear";

int main(){
  N = 1024
  M = 40
  K = 16

  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (N * M  * sizeof(float));
  file1.read((char *)input , N * M  * sizeof(float));
  Tensor<float> * input_t = new Tensor<float>(input, N, M);

  std::fstream file2(BIN_DIR + "/W.bin",std::ios::in|std::ios::binary);
  float * w  = (float *)malloc (sizeof(float) * M * K);
  file2.read((char *)w, sizeof(float) * M * K);
  Tensor<float> * W_t = new Tensor<float>(w, M, K);

  std::fstream file2(BIN_DIR + "/b.bin",std::ios::in|std::ios::binary);
  float * B  = (float *)malloc (sizeof(float) * M);
  file2.read((char *)B, sizeof(float) * M * 1);
  Tensor<float> * B_t = new Tensor<float>(B, M, 1);

  std::fstream file1(BIN_DIR + "/out.bin",std::ios::in|std::ios::binary);
  float * output = (float *)malloc (N * K  * sizeof(float));
  file1.read((char *)output , N * K  * sizeof(float));
  Tensor<float> * output_t = new Tensor<float>(output, N, K);

  std::fstream file2(BIN_DIR + "/dW.bin",std::ios::in|std::ios::binary);
  float * dw  = (float *)malloc (sizeof(float) * M * K);
  file2.read((char *)dw, sizeof(float) * M * K);
  Tensor<float> * dw_t = new Tensor<float>(dw, M, K);

  std::fstream file2(BIN_DIR + "/db.bin",std::ios::in|std::ios::binary);
  float * db  = (float *)malloc (sizeof(float) * M);
  file2.read((char *)db, sizeof(float) * M * 1);
  Tensor<float> * db_t = new Tensor<float>(db, M, 1);

  LinearLayer* l1 = new LinearLayer(W_l,B_t, N, M, K);
  Relu* l2 = new Relu(N,K);

  // Construct tensors
  // Match forward pass
  // check output at l2

  // Tensor<float> *ones = getOnes(N,K);
  // auto g = l2->backward(ones);
  // l1->backward();
  // Match backward pass
  // Check grad_w, grad_b

  std::cout << "Test ok !\n";
}
