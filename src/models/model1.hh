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
class GCNBasic{

public:
  int dim1;
  int dim2;
  int dim3;

  SageAggr *gcn1;
  SageAggr *gcn2;

  LinearLayer * ll1;
  LinearLayer * ll2;

  Relu * relu;
  // CrossEntropyLoss *loss;
  Tensor<float> out;

  GCNBasic(int dim1, int dim2, int dim3){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;

    this->gcn1 = new SageAggr(dim1);
    this->ll1 = new LinearLayer(dim1,dim2);
    this->relu = new Relu();

    this->gcn2 = new SageAggr(dim2);
    this->ll2 = new LinearLayer(dim2,dim3);
    // this->loss = new CrossEntropyLoss();
  }

  // For debugging
  GCNBasic(int dim1, int dim2, int dim3, float *W1,float *B1, float *W2, float *B2){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;

    this->gcn1 = new SageAggr(dim1);
    this->ll1 = new LinearLayer(W1,B1,dim1,dim2);
    this->relu = new Relu();

    this->gcn2 = new SageAggr(dim2);
    this->ll2 = new LinearLayer(W2,B2,dim2,dim3);
    // this->loss = new CrossEntropyLoss();
  }

// Output is used on loss functions.
// Loss for each target vertex
  Tensor<float>& forward( Tensor<int>& off_nb1, Tensor<int>& indices_nb1,
      Tensor<int>& off_nb2, Tensor<int>& indices_nb2,Tensor<float>& in, Tensor<int> &labels){
      int out_nodes = off_nb2.dim1-1;

      NNException::throwIfDeviceErrorsOccurred("Device error before forward \n");
      auto t1 = this->gcn1->forward(off_nb1,indices_nb1, in, off_nb1.dim1-1,in.dim1);
      std::cout << "First gcn layer sum " << t1.debugTensor() <<"\n";
      NNException::throwIfDeviceErrorsOccurred("gcn 1 failure \n");
      auto t2 = this->ll1->forward(t1);
      std::cout << "First linear layer sum " << t2.debugTensor() <<"\n";
      NNException::throwIfDeviceErrorsOccurred("First linear layer sum \n");
      auto t3 = this->relu->forward(t2);
      std::cout << "First relu 1 sum " << t3.debugTensor() <<"\n";
      NNException::throwIfDeviceErrorsOccurred("relu failed \n");
      auto t4 = this->gcn2->forward(off_nb2,indices_nb2,t3,off_nb2.dim1-1,t3.dim1);
      std::cout << "Second gcn layer sum " << t4.debugTensor() <<"\n";
      NNException::throwIfDeviceErrorsOccurred("Second gcn failed \n");
      Tensor<float>& t5 = this->ll2->forward(t4);
      std::cout << "Second gcn layer sum " << t5.debugTensor() <<"\n";
      // t5.debugTensor();
      return t5;
      //
      // NNException::throwIfDeviceErrorsOccurred("Catch6 \n");
      // this->out = this->loss->forward(t5,labels);
      // NNException::throwIfDeviceErrorsOccurred("Catch7 \n");
      //
      // // this->out.debugTensor();
      // // std::cout << "Catch7 \n";
      // // std::cout << "Catch7 \n";
      // return out;
  }

  void backward(Tensor<float> & grad){
    auto t5 = this->ll2->backward(grad);
    NNException::throwIfDeviceErrorsOccurred("Catch3\n");
    auto t4 = this->gcn2->backward(t5);
    NNException::throwIfDeviceErrorsOccurred("Catch4\n");
    std::cout << "REACHED HERE!!!!\n";
    auto t3 = this->relu->backward(t4);
    NNException::throwIfDeviceErrorsOccurred("Catch5\n");
    std::cout << "REACHED HERE!!!!\n";
    auto t2 = this->ll1->backward(t3);
    NNException::throwIfDeviceErrorsOccurred("Catch6\n");
    auto t1 = this->gcn1->backward(t2);
    NNException::throwIfDeviceErrorsOccurred("Catch7\n");
    std::cout << "Back7 \n";
  }

  void update(float learning_rate){
    this->ll1->update(learning_rate);
    this->ll2->update(learning_rate);
    NNException::throwIfDeviceErrorsOccurred("Update failed\n");
  }

};
