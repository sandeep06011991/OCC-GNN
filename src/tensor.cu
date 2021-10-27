#include "tensor.hh"
#include "nn_exception.hh"
#include <random>
#include <assert.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include<iostream>
#include <stdio.h>      /* printf */
#include <math.h>
#include <algorithm>



// template<typename T>
__global__
void gradient_update(float * x, float * x_grad, int size, float learning_rate){
  int globalIdx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(globalIdx < size){
    x[globalIdx] = x[globalIdx] - (x_grad[globalIdx] * learning_rate );
  }
}


template<typename T>
void Tensor<T>::update(float learning_rate, Tensor<T> * grad){
  assert(this->dim1 == grad->dim1);
  assert(this->dim2 == grad->dim2);

  int totalSize = this->dim1 * this->dim2;
  int noBlocks = (totalSize  + 255)/256;
  int noThreads = 256;
  gradient_update<<<noBlocks,noThreads>>>((float *)this->data_device,(float *)grad->data_device,totalSize,learning_rate);

}

template<typename T>
Tensor<T>::Tensor(int dim1, int dim2){
  this->dim1 = dim1;
  this->dim2 = dim2;
  assert(this->dim1 > 0);
  assert(this->dim2 > 0);

  this->allocateMemory();
  NNException::throwIfDeviceErrorsOccurred("cudaalloc data failed\n");
  cudaDeviceSynchronize();
  NNException::throwIfDeviceErrorsOccurred("cudaalloc data failed\n");
  // this->debugTensor();
}

template<typename T>
Tensor<T>::Tensor(T* data, int dim1, int dim2){
  this->data_host = data;
  this->dim1 = dim1;
  this->dim2 = dim2;
  this->allocateMemory();
  this->copyHostToDevice();
  this->data_host = nullptr;
  cudaDeviceSynchronize();
}

template<typename T>
void Tensor<T>::allocateMemory(){
   // std::cout << "allocate " << dim1 *dim2 <<"\n";
   cudaMalloc(&this->data_device, dim1 * dim2 * sizeof(T));
   NNException::throwIfDeviceErrorsOccurred("memory allocation failed");
}

template<typename T>
void Tensor<T>::copyHostToDevice(){
   assert(this->data_device !=nullptr);
   assert(this->data_host !=nullptr);
   cudaMemcpy(this->data_device,this->data_host,dim1 * dim2 * sizeof(T), cudaMemcpyHostToDevice);
   NNException::throwIfDeviceErrorsOccurred("memory copy failed");
}

template<typename T>
void Tensor<T>::copyDeviceToHost(){
  if(this->data_host == nullptr){
     this->data_host = (T *)malloc(sizeof(T) * dim1 * dim2);
   }
   assert(this->data_device !=nullptr);
   assert(this->data_host !=nullptr);
   cudaMemcpy(this->data_host,this->data_device,dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   NNException::throwIfDeviceErrorsOccurred("memory copy failed");
}

template<typename T>
T Tensor<T>::debugTensor(){
  this->copyDeviceToHost();
  // for(int i=0;i<4;i++){
  //   for(int j=0;j<4;j++){
  //     std::cout << this->data_host[i*this->dim2+j] << " ";
  //   }
  //   std::cout << "\n";
  // }
  T s = 0;
  for(int i=0;i<this->dim1;i++){
    for(int j=0;j<this->dim2;j++){
      s += this->data_host[i*this->dim2+j] ;
    }
  }
  // std::cout << "SUM " << s <<"\n";
  return s;
}
float * allocate_random(int size,int dim){
  float stdv = 1. / sqrt(dim);
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-stdv, stdv);
  float * data = (float *)malloc(sizeof(float)*size);
  for(int i=0;i<size;i++){
    data[i] =  dis(gen);

  }
  return data;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
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
                std::cout << "\33[31mErrorjjj: " << cublasGetErrorString(success) << "\33[0m\n";

        }
      cublasDestroy(handle);
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

        }
    cublasDestroy(handle);
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

        }
    cublasDestroy(handle);
}

Tensor<float> * allocate_ones(int dim1, int dim2){
   float * data_host = (float *)malloc(dim1 * dim2 * sizeof(float));
   for(int i=0;i<dim1*dim2;i++){
     data_host[i] = 1;
   }

   Tensor<float> * ret =  new Tensor<float>(data_host,dim1, dim2);
   ret->data_host = nullptr;
   free(data_host);
   return ret;
}

bool approx_equal(int a,int b){
  return a==b;
}

bool approx_equal(float a,float b){
  // std::cout << a << " " << b <<"\n";
  // std::cout << "doffer" <<b - a<<"\n";
  if(a==b)return true;
  if(a>b){
    return (a-b)/a < .0001;
  }
  return (b-a)/a < .00001;
}
