#include "util/tensor.hh"
#include "nn_exception.hh"
#include <random>
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>      /* printf */
#include <math.h>
#include <algorithm>

// template<typename T>
__global__ void gradient_update(float * x, float * x_grad, int size, float learning_rate){
  int globalIdx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(globalIdx < size){
    x[globalIdx] = x[globalIdx] - (x_grad[globalIdx] * learning_rate );
  }
}


template<typename T>
void Tensor<T>::update(float learning_rate, Tensor<T> * grad){
  assert(this->s.dim1 == grad->s.dim1);
  assert(this->s.dim2 == grad->s.dim2);

  int totalSize = this->s.dim1 * this->s.dim2;
  int noBlocks = (totalSize  + 255)/256;
  int noThreads = 256;
  cudaSetDevice(device_id);
  gradient_update<<<noBlocks,noThreads>>>((float *)this->data_device,(float *)grad->data_device,totalSize,learning_rate);

}


template<typename T>
Tensor<T>::Tensor(Shape s, int device_id){
  this->s = s;
  this->device_id = device_id;
  assert(device_id >= 0);
  cudaSetDevice(device_id);
  cudaMalloc(&this->data_device, s.dim1 * s.dim2 *sizeof(T));
  NNException::throwIfDeviceErrorsOccurred("cudamlloc data failed\n");
  cudaDeviceSynchronize();
}

template<typename T>
Tensor<T>::Tensor(T* data, Shape s, int device_id){
  this->s = s;
  this->device_id = device_id;
  assert(device_id >= 0);
  cudaSetDevice(device_id);
  cudaMalloc(&this->data_device, s.dim1 * s.dim2 *sizeof(T));
  NNException::throwIfDeviceErrorsOccurred("cudamalloc data failed\n");
  cudaMemcpy(this->data_device, data, s.dim1 * s.dim2 *sizeof(T), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

template<typename T>
Tensor<T>::Tensor(Tensor<T> &src, int device_id){
  assert(src.device_id != device_id);
  this->s = src.s;
  this->device_id = device_id;
  assert(device_id >= 0);
  cudaSetDevice(device_id);
  cudaMalloc(&this->data_device, this->s.dim1 * this->s.dim2 *sizeof(T));
  NNException::throwIfDeviceErrorsOccurred("cudamalloc data failed\n");
  cudaMemcpy(this->data_device, src.data_device, s.dim1 * s.dim2 *sizeof(T), cudaMemcpyDeviceToDevice);
  NNException::throwIfDeviceErrorsOccurred("Memcpy data failed\n");
  cudaDeviceSynchronize();
}

template<typename T>
Tensor<T>::Tensor(Tensor<T> *src, int device_id){
  assert(src->device_id != device_id);
  this->s = src->s;
  this->device_id = device_id;
  assert(device_id >= 0);
  cudaSetDevice(device_id);
  cudaMalloc(&this->data_device, this->s.dim1 * this->s.dim2 *sizeof(T));
  NNException::throwIfDeviceErrorsOccurred("cudamalloc data failed\n");
  cudaMemcpy(this->data_device, src->data_device, s.dim1 * s.dim2 *sizeof(T), cudaMemcpyDeviceToDevice);
  NNException::throwIfDeviceErrorsOccurred("Memcpy data failed\n");
  cudaDeviceSynchronize();
}

template<typename T>
void Tensor<T>::viewTensor(){
  T * host = (T *)malloc(sizeof(T) * this->s.dim1 * this->s.dim2);
  cudaSetDevice(device_id);
  cudaMemcpy(host, this->data_device,  s.dim1 * s.dim2 *sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int ii = std::min(4, this->s.dim1);
  int jj = std::min(4, this->s.dim2);

  // std::cout << "On device" << this->device_id <<"\n";
  // for(int i=0;i<ii;i++){
  //    for(int j=0;j<jj;j++){
  //      std::cout << host[i*this->s.dim2+j] << " ";
  //    }
  //    std::cout << "\n";
  //  }
  float s = 0;
  for(int i=0; i< (this->s.dim1);i++){
    s = s + host[i * this->s.dim2];
  }
  std::cout << "Total sum on gpu:" <<device_id <<"sum :"<< s <<"\n";
  free(host);

}
template<typename T>
void Tensor<T>::copyTensorToCPUMemory(T* cpu_mem){
  cudaSetDevice(device_id);
  cudaMemcpy(cpu_mem, this->data_device,  s.dim1 * s.dim2 *sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

// template<typename T>
// void Tensor<T>::allocateMemory(){
//    // std::cout << "allocate " << dim1 *dim2 <<"\n";
//    cudaMalloc(&this->data_device, dim1 * dim2 * sizeof(T));
//    NNException::throwIfDeviceErrorsOccurred("memory allocation failed");
// }
//
// template<typename T>
// void Tensor<T>::copyHostToDevice(){
//    assert(this->data_device !=nullptr);
//    assert(this->data_host !=nullptr);
//    cudaMemcpy(this->data_device,this->data_host,dim1 * dim2 * sizeof(T), cudaMemcpyHostToDevice);
//    NNException::throwIfDeviceErrorsOccurred("memory copy failed");
// }
//
// template<typename T>
// void Tensor<T>::copyDeviceToHost(){
//   if(this->data_host == nullptr){
//      this->data_host = (T *)malloc(sizeof(T) * dim1 * dim2);
//    }
//    assert(this->data_device !=nullptr);
//    assert(this->data_host !=nullptr);
//    cudaMemcpy(this->data_host,this->data_device,dim1 * dim2 * sizeof(T), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
//    NNException::throwIfDeviceErrorsOccurred("memory copy failed");
// }
//
// template<typename T>
// T Tensor<T>::debugTensor(){
//   this->copyDeviceToHost();
//   // for(int i=0;i<4;i++){
//   //   for(int j=0;j<4;j++){
//   //     std::cout << this->data_host[i*this->dim2+j] << " ";
//   //   }
//   //   std::cout << "\n";
//   // }
//   T s = 0;
//   for(int i=0;i<this->dim1;i++){
//     for(int j=0;j<this->dim2;j++){
//       s += this->data_host[i*this->dim2+j] ;
//     }
//   }
//   // std::cout << "SUM " << s <<"\n";
//   return s;
// }
// float * allocate_random(int size,int dim){
//   float stdv = 1. / sqrt(dim);
//   std::random_device rd;  // Will be used to obtain a seed for the random number engine
//   std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
//   std::uniform_real_distribution<> dis(-stdv, stdv);
//   float * data = (float *)malloc(sizeof(float)*size);
//   for(int i=0;i<size;i++){
//     data[i] =  dis(gen);
//
//   }
//   return data;
// }
//
//
// Tensor<float> * allocate_ones(int dim1, int dim2){
//    float * data_host = (float *)malloc(dim1 * dim2 * sizeof(float));
//    for(int i=0;i<dim1*dim2;i++){
//      data_host[i] = 1;
//    }
//
//    Tensor<float> * ret =  new Tensor<float>(data_host,dim1, dim2);
//    ret->data_host = nullptr;
//    free(data_host);
//    return ret;
// }
