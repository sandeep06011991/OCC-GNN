#include <iostream>
#include <string>
#include "tensor.hh"
#include <fstream>
#include <assert.h>

std::string BIN_DIR = "/homes/spolisetty/data/tests/bce";

int main(){

  int X = 19717;
  int D = 3;
  X = 1024;
  D = 5;
  cudaSetDevice(0);
  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (X * D  * sizeof(float));
  file1.read((char *)input , X * D * sizeof(float));
  Tensor<float> * input_t = new Tensor<float>(input, X, D);
  input_t->debugTensor();
  input_t->viewTensor();
  cudaDeviceSynchronize();
  std::cout << "Hello world!\n";
}
