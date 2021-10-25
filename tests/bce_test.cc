#include <iostream>
#include <string>
#include "layers/loss.hh"
#include "tensor.hh"
#include <fstream>
#include <assert.h>

std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/bce";

int main(){

  int X = 19717;
  int D = 3;
  // X = 1024;
  // D = 32;

  std::fstream file1(BIN_DIR + "/input.bin",std::ios::in|std::ios::binary);
  float * input = (float *)malloc (X * D  * sizeof(float));
  file1.read((char *)input , X * D * sizeof(float));
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
  float * grad  = (float *)malloc (sizeof(float) * X * D);
  file4.read((char *)grad, sizeof(float) * D * X);
  Tensor<float> * grad_t = new Tensor<float>(grad, X, D);

  CrossEntropyLoss* loss = new CrossEntropyLoss(X,D);

  std::cout << "Beginning forward Pass !!\n";

  Tensor<float>& loss_t = loss->forward(*input_t,*labels_t);
  loss_t.copyDeviceToHost();
  float s = 0;
  int size = loss_t.dim1 * loss_t.dim2;
  for(int i=0;i<size;i++){
    s += loss_t.data_host[i];
  }
  // assert(s/size - out < .0001);
  std::cout << "Total loss predicted :" << loss_t.debugTensor()/loss_t.dim1 << "actual " << out << "\n";

  Tensor<float>& dgrad = loss->backward(*labels_t);
  std::cout << "backward pass ok!\n";
  float a = grad_t->debugTensor();
  float b = dgrad.debugTensor();
  std::cout << "total dgrad sum " << a <<"\n";
  std::cout << "total dgrad sum " << b << "\n";
  assert(approx_equal(a,b));
  grad_t->viewTensor();
  dgrad.viewTensor();

  for(int i=0;i<dgrad.dim1*dgrad.dim2;i++){
    float t = dgrad.data_host[i] - grad_t->data_host[i];
    if(t>1e-10){
      std::cout << i <<":" << t  << ":" \
    << dgrad.data_host[i] << ":" << grad_t->data_host[i] <<"\n";}
  }


  std::cout << "Total target float: " << b <<"\n";


  std::cout << "Test ok !\n";
}
