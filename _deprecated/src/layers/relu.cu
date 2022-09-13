#include "layers/relu.hh"
#include <iostream>

__global__ void cuda_forward(float * in, float * out, int size){
  int globalIdx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(globalIdx < size){
    float t = in[globalIdx];
    if(t <= 0){
      t = 0;
    }
    out[globalIdx] = t;
  }
}

__global__ void cuda_backward(float * in_g, float * out, float * out_g, int size){
  int globalIdx = threadIdx.x + (blockIdx.x * blockDim.x);
  if(globalIdx < size){
    float t = out[globalIdx];
    float r = in_g[globalIdx];
    if(t <= 0){
      r = 0;
    }
    out_g[globalIdx] = r;
  }
}

Relu::Relu(){
  this->dim1 = -1;
  this->dim2 = -1;
}

Relu::Relu(int dim1, int dim2){
    this->dim1 = dim1;
    this->dim2 = dim2;
}

Tensor<float>& Relu::forward(Tensor<float>& X){
  this->dim1 = X.dim1;
  this->dim2 = X.dim2;
  if(this->out != nullptr){
    this->out->clearTensor();
    this->d_out->clearTensor();
    delete this->out;
    delete this->d_out;
  }

  this->out = new Tensor<float>(this->dim1,this->dim2);
  this->d_out = new Tensor<float>(this->dim1,this->dim2);


  int size = this->dim1 * this->dim2;
  int noBlocks = (size  + 255)/256;
  int noThreads = 256;
  cuda_forward<<<noBlocks,noThreads>>>(X.data_device,out->data_device,size);
  cudaDeviceSynchronize();
  return *this->out;
  // return *(new Tensor<float>(1,1));
}

Tensor<float>& Relu::backward(Tensor<float>& grad_x){
  std::cout << this->dim1 << " " << this->dim2 <<"\n";
  int size = this->dim1 * this->dim2;
  int noBlocks = (size  + 255)/256;
  int noThreads = 256;
  cuda_backward<<<noBlocks,noThreads>>>(grad_x.data_device, out->data_device,
            d_out->data_device, size);
  cudaDeviceSynchronize();
  return *d_out;
}
