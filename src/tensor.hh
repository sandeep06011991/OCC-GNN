#pragma once
#include<iostream>
#include <cuda_runtime.h>
#include "nn_exception.hh"
#include <assert.h>
// Currently a matrix(i.e 2D),
// extend later for more complex shapes.
// Tensor only guarantees space exists on gpu.
template<typename T>
class Tensor {

private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:

	T* data_device = nullptr;
	T* data_host = nullptr;
  int dim1;
  int dim2;

  // T* grad_device = nullptr;
  bool has_grad;

	Tensor(int x_dim = 1,int y_dim = 1);
  Tensor(T* data, int  x_dim, int y_dim);

	void allocateMemory();

	void copyHostToDevice();
	void copyDeviceToHost();

	// float& operator[](const int index);
	// const float& operator[](const int index) const;
  void verify(T* correct_data);
  T debugTensor();

  void update(float learning_rate,Tensor * grad);

  void viewTensor(){
    // assert(dim1 > 2);
    // assert(dim2 > 2);
    int J = std::min(dim2,3);
    for(int i=0;i<3;i++){
      for(int j=0;j<J;j++){
        std::cout << data_host[i * dim2  + j] <<" ";
      }
      std::cout << "\n ";

    }
    std::cout <<"\n";
  }

  void cleanUpTensor(){
    NNException::throwIfDeviceErrorsOccurred("Error before attempting to free\n");
    if(data_device !=nullptr){
      cudaError_t error = cudaFree(data_device);
      // std::cout <<"free " << dim1 * dim2 <<"\n";
      NNException::throwIfDeviceErrorsOccurred("cudaFree data failed\n");
    }
    if(data_host!=nullptr){
      free(data_host);
    }
    // if(grad_device !=nullptr){
    //   cudaFree(grad_device);
    //   NNException::throwIfDeviceErrorsOccurred("cudaFree of grad failed\n");
    // }
  }
};

template class Tensor<float>;
template class Tensor<int>;


bool approx_equal(int a,int b);

bool approx_equal(float a,float b);


Tensor<float> * allocate_ones(int dim1, int dim2);

float * allocate_random(int size,int dim);

bool tensor_equal(Tensor<float> &a, Tensor<float> &b);

void mat_mul_a_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);

void mat_mul_a_b_t(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);

void mat_mul_a_t_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);
