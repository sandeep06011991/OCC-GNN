#include <iostream>
#include <string>
#include "layers/loss.hh"
#include "tensor.hh"
#include <fstream>
#include <assert.h>

std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/bce";

int main(){
  int X = 1024;
  int D = 8;

  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (X * 8  * sizeof(float));
  file1.read((char *)input , X * 8  * sizeof(float));
  Tensor<float> * input_t = new Tensor<float>(input, X, D);

  std::fstream file2(BIN_DIR + "/target.bin",std::ios::in|std::ios::binary);
  int * target  = (int *)malloc (sizeof(int) * X);
  file2.read((char *)target, sizeof(int) * X);
  Tensor<int> * labels_t = new Tensor<int>(target, X, 1);

  float out;
  std::fstream file3(BIN_DIR + "/out.bin",std::ios::in|std::ios::binary);
  file3.read((char *)&out, sizeof(float));
  std::cout << "Target total loss" << out <<"\n";

  std::fstream file4(BIN_DIR + "/grad.bin",std::ios::in|std::ios::binary);
  float * grad  = (float *)malloc (sizeof(float) * X * 8);
  file4.read((char *)grad, sizeof(float) * 8 * X);
  Tensor<float> * grad_t = new Tensor<float>(grad, X, D);

  CrossEntropyLoss* loss = new CrossEntropyLoss(X,D);

  std::cout << "Beginning forward Pass !!\n";

  Tensor<float>& loss_t = loss->lossForward(*input_t,*labels_t);
  loss_t.copyDeviceToHost();
  float s = 0;
  int size = loss_t.dim1 * loss_t.dim2;
  for(int i=0;i<size;i++){
    s += loss_t.data_host[i];
  }
  assert(s/size - out < .01);
  // loss_t.debugTensor();

  Tensor<float>& dgrad = loss->lossBackward(*labels_t);
  dgrad.debugTensor();
  
  for(int i=0;i<dgrad.dim1*dgrad.dim2;i++){
    float t = dgrad.data_host[i] - grad_t->data_host[i];
    if(t>.0001){
      std::cout << i <<":" << t  << ":" \
    << dgrad.data_host[i] << ":" << grad_t->data_host[i] <<"\n";}
  }
  // std::cout << "Total target float: " << s/size <<"\n";


  std::cout << "Test ok !\n";
}
