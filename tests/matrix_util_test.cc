#include <iostream>
#include <string>
#include "tensor.hh"
#include <fstream>


std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/matrix";

int main(){
  int M = 1000;
  int N = 156;
  int K = 211;
  // int M = 8;
  // int N = 8;
  // int K = 8;

  std::fstream file1(BIN_DIR + "/A.bin",std::ios::in|std::ios::binary);
  float * A = (float *)malloc (M * N  * sizeof(float));
  file1.read((char *)A , N * M  * sizeof(float));
  Tensor<float> * A_t = new Tensor<float>(A, M, N);

  std::fstream file2(BIN_DIR + "/B.bin",std::ios::in|std::ios::binary);
  float * B  = (float *)malloc (sizeof(float) * N * K);
  file2.read((char *)B, sizeof(float) * N * K);
  // for(int i=0;i<N*K;i++){
  //   B[i] = 1;
  // }

  Tensor<float> * B_t = new Tensor<float>(B, N, K);


  std::fstream file3(BIN_DIR + "/C.bin",std::ios::in|std::ios::binary);
  float * C  = (float *)malloc (sizeof(float) * M * K);
  file3.read((char *)C, sizeof(float) *  M * K);
  Tensor<float> * C_t = new Tensor<float>(C, M , K);

  std::fstream file4(BIN_DIR + "/dA.bin",std::ios::in|std::ios::binary);
  float * dA = (float *)malloc (N * M  * sizeof(float));
  file4.read((char *)dA , N * M * sizeof(float));
  Tensor<float> * dA_t = new Tensor<float>(dA, M, N);

  std::fstream file5(BIN_DIR + "/dW.bin",std::ios::in|std::ios::binary);
  float * dW = (float *)malloc ( N * K  * sizeof(float));
  file5.read((char *)dW , N * K * sizeof(float));
  Tensor<float> * dW_t = new Tensor<float>(dW, N, K);
  Tensor<float> * dW_calc = new Tensor<float>(N,K);

  Tensor<float> * C_calc = new Tensor<float>(M,K);
  Tensor<float> * dA_calc = new Tensor<float>(M,N);

  mat_mul_a_b(*A_t, false, *B_t, false, *C_calc);
  mat_mul_a_b_t(*C_t,false,*B_t, false, *dA_calc);
  mat_mul_a_t_b(*A_t,false,*C_t,false, *dW_calc);
  dW_t->debugTensor();
  dW_calc->debugTensor();
  dA_t->debugTensor();
  dA_calc->debugTensor();
  C_calc->debugTensor();
  C_t->debugTensor();
  std::cout << "haha \n";

}
