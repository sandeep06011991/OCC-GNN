#include "tensor.hh"
#include "nn_exception.hh"

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
