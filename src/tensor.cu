#include "tensor.hh"
#include "nn_exception.hh"
#include <random>
#include <assert.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

template<typename T>
Tensor<T>::Tensor( int dim1, int dim2){
  this->dim1 = dim1;
  this->dim2 = dim2;
  this->allocateMemory();
  cudaDeviceSynchronize();
}

template<typename T>
Tensor<T>::Tensor(T* data, int dim1, int dim2){
  this->data_host = data;
  this->dim1 = dim1;
  this->dim2 = dim2;
  this->allocateMemory();
  this->copyHostToDevice();
  cudaDeviceSynchronize();
}

template<typename T>
void Tensor<T>::allocateMemory(){
   cudaMalloc(&this->data_device, dim1 * dim2 * sizeof(T));
   NNException::throwIfDeviceErrorsOccurred("memory allocation failed");
}

template<typename T>
void Tensor<T>::copyHostToDevice(){
   cudaMemcpy(this->data_device,this->data_host,dim1 * dim2 * sizeof(T), cudaMemcpyHostToDevice);
   NNException::throwIfDeviceErrorsOccurred("memory copy failed");
}

template<typename T>
void Tensor<T>::copyDeviceToHost(){
  if(this->data_host == nullptr){
     this->data_host = (T *)malloc(sizeof(T) * dim1 * dim2);
   }
   cudaMemcpy(this->data_host,this->data_device,dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   NNException::throwIfDeviceErrorsOccurred("memory copy failed");
}

template<typename T>
void Tensor<T>::debugTensor(){
  this->copyDeviceToHost();
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      std::cout << this->data_host[i*this->dim2+j] << " ";
    }
    std::cout << "\n";
  }
  T s = 0;
  for(int i=0;i<this->dim1;i++){
    for(int j=0;j<this->dim2;j++){
      s += this->data_host[i*this->dim2+j] ;
    }
  }
  std::cout << "SUM " << s <<"\n";
}
float * allocate_random(int size){
  float * data = (float *)malloc(sizeof(float)*size);
  for(int i=0;i<size;i++){
    data[i] = rand();
  }
  return data;
}

// ,mat mul in row major format
void mat_mul_a_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& C){
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t success= cublasSgemm( handle, CUBLAS_OP_N , CUBLAS_OP_N ,C.dim2, C.dim1
       , B.dim1 , \
       &alpha,   B.data_device, B.dim2, A.data_device, A.dim2 ,
        &beta , C.data_device, C.dim2);
    if ( success != CUBLAS_STATUS_SUCCESS){
                std::cout << "\33[31mError: " << success << "\33[0m\n";
            cublasDestroy(handle);
        }
}

// ,mat mul in row major format
void mat_mul_a_t_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& C){
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t success= cublasSgemm( handle,  CUBLAS_OP_N , CUBLAS_OP_T  ,C.dim2, C.dim1
       , B.dim1  , \
       &alpha,   B.data_device, B.dim2, A.data_device, A.dim2 ,
        &beta , C.data_device, C.dim2);
    if ( success != CUBLAS_STATUS_SUCCESS){
                std::cout << "\33[31mError: " << success << "\33[0m\n";
            cublasDestroy(handle);
        }
}

// ,mat mul in row major format
void mat_mul_a_b_t(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& C){
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t success= cublasSgemm( handle, CUBLAS_OP_T , CUBLAS_OP_N ,C.dim2, C.dim1
       , B.dim2 , \
       &alpha,   B.data_device, B.dim2, A.data_device, A.dim2 ,
        &beta , C.data_device, C.dim2);
    if ( success != CUBLAS_STATUS_SUCCESS){
                std::cout << "\33[31mError: " << success << "\33[0m\n";
            cublasDestroy(handle);
        }
}
