#include <iostream>
#include <string>
#include "layers/loss.hh"
#include "util/tensor.hh"
#include <fstream>
#include <assert.h>

std::string BIN_DIR = "/home/spolisetty/data/tests/bce";

int main(){

  int X = 19717;
  int D = 3;
  X = 1024;
  D = 5;
  int device_id = 0;
  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (X * D  * sizeof(float));
  file1.read((char *)input , X * D * sizeof(float));
  Tensor<float> * input_t = new Tensor<float>(input, Shape(X, D),device_id);

  std::fstream file2(BIN_DIR + "/target.bin",std::ios::in|std::ios::binary);
  int * target  = (int *)malloc (sizeof(int) * X);
  file2.read((char *)target, sizeof(int) * X);
  Tensor<int> * labels_t = new Tensor<int>(target, Shape(X, 1),device_id);

  float out;
  std::fstream file3(BIN_DIR + "/out.bin",std::ios::in|std::ios::binary);
  file3.read((char *)&out, sizeof(float));
  std::cout << "Target total loss" << out <<"\n";

  std::fstream file4(BIN_DIR + "/grad.bin",std::ios::in|std::ios::binary);
  float * grad  = (float *)malloc (sizeof(float) * X * D);
  file4.read((char *)grad, sizeof(float) * D * X);
  Tensor<float> * grad_t = new Tensor<float>(grad, Shape(X, D),device_id);

  CrossEntropyLoss* loss = new CrossEntropyLoss(X,D,device_id);

  std::cout << "Beginning forward Pass !!\n";

  Tensor<float>& loss_t = loss->forward(*input_t,*labels_t);
  int size = loss_t.s.dim1 * loss_t.s.dim2;
  float * cpu_loss = (float *)malloc(sizeof(float) * size);
  loss_t.copyTensorToCPUMemory(cpu_loss);
  float s = 0;
  for(int i=0;i<size;i++){
    s += cpu_loss[i];
  }
  // assert(s/size - out < .0001);
  std::cout << "Total loss predicted :" << s/loss_t.s.dim1 << "actual " << out << "\n";

  Tensor<float>& dgrad = loss->backward(*labels_t);
  std::cout << "backward pass ok!\n";
  // float a = grad_t->debugTensor();
  // float b = dgrad.debugTensor();
  // std::cout << "total dgrad sum " << a <<"\n";
  // std::cout << "total dgrad sum " << b << "\n";
  // assert(approx_equal(a,b));
  grad_t->viewTensor();
  dgrad.viewTensor();

  // for(int i=0;i<dgrad.dim1*dgrad.dim2;i++){
  //   float t = abs(dgrad.data_host[i] - grad_t->data_host[i]);
  //   // assert(dgrad.data_host[i]*10000 * grad_t->data_host[i]*10000 > 0);
  //   if(t>1e-10){
  //     std::cout << i <<":" << t  << ":" \
  //   << dgrad.data_host[i] << ":" << grad_t->data_host[i] <<"\n";
  //   }
  // }
  std::cout << "Test ok !\n";
}
