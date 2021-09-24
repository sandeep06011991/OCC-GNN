#include "tensor.hh"
#include "nn_exception.hh"
#include <random>

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
    for(int j=0;j<this->dim2;j++){
      std::cout << this->data_host[i*this->dim1+j] << " ";
    }
    std::cout << "\n";
  }
}
float * allocate_random(int size){
  float * data = (float *)malloc(sizeof(float)*size);
  for(int i=0;i<size;i++){
    data[i] = rand();
  }
  return data;
}
