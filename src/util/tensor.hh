#pragma once
#include  <iostream>
#include  <cuda_runtime.h>
#include  "nn_exception.hh"
#include  <assert.h>

struct Shape{
  int dim1;
  int dim2;
  int dim3;

  Shape(){}

  Shape(int dim1, int dim2){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = -1;
  }


  bool operator ==(Shape &s){
    if(dim1!=s.dim1)return false;
    if(dim2!=s.dim2)return false;
    if(dim3!=s.dim3)return false;
    return true;
  }
};

// Currently a matrix(i.e 2D),
// extend later for more complex shapes.
// Tensor only guarantees space exists on gpu.
template<typename T>
class Tensor {

private:
  // if device_id = -1, indicated CPU pointer.
  int device_id;
  T* data_device;


	// void allocateCudaMemory();
	// void allocateHostMemory();
	// void allocateMemory();
	// void copyHostToDevice();
	// void copyDeviceToHost();

public:

  Shape s;
  Tensor(Shape s, int device);
  Tensor(T* data, Shape s, int device);
  Tensor(Tensor<T>& t,int device);

	void verify(T* correct_data);
  void  debugTensor();

  void update(float learning_rate,Tensor * grad);

  // void viewTensor(){
  //   int J = std::min(dim2,3);
  //   for(int i=0;i<10;i++){
  //     for(int j=0;j<J;j++){
  //       std::cout << data_host[i * dim2  + j] <<" ";
  //     }
  //     std::cout << "\n ";
  //
  //   }
  //   std::cout <<"\n";
  // }

  void clearTensor(){
    NNException::throwIfDeviceErrorsOccurred("Error before attempting to free\n");
    if(data_device !=nullptr){
      cudaError_t error = cudaFree(data_device);
      // std::cout <<"free " << dim1 * dim2 <<"\n";
      NNException::throwIfDeviceErrorsOccurred("cudaFree data failed\n");
    }
  //   if(data_host!=nullptr){
  //     free(data_host);
  //   }
  }
};

template class Tensor<float>;
template class Tensor<int>;
//
// bool approx_equal(int a,int b);
//
// bool approx_equal(float a,float b);
//
Tensor<float> * allocate_ones(int dim1, int dim2);
//
// float * allocate_random(int size,int dim);

bool tensor_equal(Tensor<float> &a, Tensor<float> &b);
