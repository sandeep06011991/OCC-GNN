#pragma once
#include "tensor.hh"
#include "gnn/sage.hh"
#include "layers/linear.hh"
#include "layers/relu.hh"
#include "layers/loss.hh"
#include <iostream>
#include <nn_exception.hh>

// 2 Layer GCN which relies on 2 hop aggregation.
// Uses a single gpu setting.
class TwoLayerModel{

public:
  int dim1;
  int dim2;
  int dim3;

  LinearLayer * ll1;
  LinearLayer * ll2;

  Relu * relu;
  // CrossEntropyLoss *loss;
  Tensor<float> out;

  TwoLayerModel(int dim1, int dim2, int dim3){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;

    this->ll1 = new LinearLayer(dim1,dim2);
    this->relu = new Relu();

    this->ll2 = new LinearLayer(dim2,dim3);
    // this->loss = new CrossEntropyLoss();
  }

  // For debugging
  TwoLayerModel(int dim1, int dim2, int dim3, float *W1,float *B1, float *W2, float *B2){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;

    this->ll1 = new LinearLayer(W1,B1,dim1,dim2);
    this->relu = new Relu();

    this->ll2 = new LinearLayer(W2,B2,dim2,dim3);
    // this->loss = new CrossEntropyLoss();
  }

// Output is used on loss functions.
// Loss for each target vertex
  Tensor<float>& forward(Tensor<float>& in){
      auto t2 = this->ll1->forward(in);
      // std::cout << "First linear layer sum " << t2.debugTensor() <<"\n";
      NNException::throwIfDeviceErrorsOccurred("First linear layer sum \n");

      auto t3 = this->relu->forward(t2);
      // // std::cout << "First relu 1 sum " << t3.debugTensor() <<"\n";
      NNException::throwIfDeviceErrorsOccurred("relu failed \n");
      // std::cout << "check point before !\n";
      Tensor<float>& t5 = this->ll2->forward(t3);
      // std::cout << "check point after !\n";

      // std::cout << "Second gcn layer sum " << t5.debugTensor() <<"\n";
      return t5;
  }

  void backward(Tensor<float> & grad){

    auto t5 = this->ll2->backward(grad);
    NNException::throwIfDeviceErrorsOccurred("ll2 grad fail\n");
    auto t3 = this->relu->backward(t5);
    std::cout << "Disappearing gradients\n";
    t3.debugTensor();
    t3.viewTensor();
    NNException::throwIfDeviceErrorsOccurred("relu grad fail\n");
    auto t2 = this->ll1->backward(t3);
    NNException::throwIfDeviceErrorsOccurred("ll1 grad fail\n");
  }

  void update(float learning_rate){
    this->ll1->update(learning_rate);
    this->ll2->update(learning_rate);
    NNException::throwIfDeviceErrorsOccurred("Update failed\n");
  }

};
