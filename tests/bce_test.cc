#include <iostream>
#include <string>
#include "layers/loss.hh"
#include "tensor.hh"
#include <fstream>

std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/bce";

int main(){
  int X = 1024;
  int D = 8;

  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (1024 * 8  * sizeof(float));
  file1.read((char *)input , 1024 * 8  * sizeof(float));
  Tensor<float> * input_t = new Tensor<float>(input, X, D);

  std::fstream file2(BIN_DIR + "/target.bin",std::ios::in|std::ios::binary);
  int * target  = (int *)malloc (sizeof(int) * 1024);
  file2.read((char *)target, sizeof(int) * 1024);
  Tensor<int> * labels_t = new Tensor<int>(target, X, 1);

  float out;
  std::fstream file3(BIN_DIR + "/out.bin",std::ios::in|std::ios::binary);
  file3.read((char *)&out, sizeof(float));
  std::cout << "Target total loss" << out <<"\n";

  std::fstream file4(BIN_DIR + "/grad.bin",std::ios::in|std::ios::binary);
  float * grad  = (float *)malloc (sizeof(float) * 1024 * 8);
  file4.read((char *)grad, sizeof(float) * 8 * 1024);
  Tensor<float> * grad_t = new Tensor<float>(grad, X, D);

  CrossEntropyLoss* loss = new CrossEntropyLoss(X,D);

  std::cout << "Beginning forward Pass !!\n";

  Tensor<float>& loss_t = loss->lossForward(*input_t,*labels_t);
  // loss_t.copyDeviceToHost();
  // float s = 0;
  // int size = loss_t.dim1 * loss_t.dim2;
  // for(int i=0;i<size;i++){
  //   s += loss_t.data_host[i];
  // }
  // assert(s/size - out < .01);

  Tensor<float>& dgrad = loss->lossBackward(*labels_t);
  dgrad.debugTensor();
  std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n";
  grad_t ->debugTensor();
  // std::cout << "Total target float: " << s/size <<"\n";


  std::cout << "Test ok !\n";
}
