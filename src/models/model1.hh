#pragma once
#include "tensor.hh"
#include "gnn/sage.hh"
#include "layers/linear.hh"
#include "layers/relu.hh"


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

  Relu * rl1;
  Tensor<float> * out = nullptr;

  GCNBasic(int dim1, int dim2, int dim3){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->dim3 = dim3;
    // this->gcn1 = new SageAggr(dim1);
    // this->ll1 = new LinearLayer(dim1,dim2);
    // this->relu1 = new Relu();
    // this->gcn2 = new SageAggr(dim2);
    // this->ll2 = new LinearLayer(dim2,dim3);
  }

// Output is used on loss functions.
  Tensor<float>& forward(Tensor<float>& in, Tensor<int>& off_nb1, Tensor<int>& indices_nb1,
        Tensor<int>& off_nb2, Tensor<int>& indices_nb2){
        int out_nodes = off_nb2.dim1;
        if(out != nullptr){
           delete(out);
           out = new Tensor<float>(out_nodes,dim3);
        }
      // auto t1 = this->gcn1->forward(off_nb1,indices_nb1,in);
      // auto t2 = this->ll1->forward(t1);
      // auto t3 = this->relu->forward(t2);
      // auto t4 = this->gcn2->forward(off_nb2,indices_nb2,t3);
      // auto t5 = this->ll2->forward(t3);
      return *out;
  }

  void backward(Tensor<float> & inGrad){
    // auto t5 = this->ll2->backward(inGrad);
    // auto t4 = this->gcn2->backward(t5);
    // auto t3 = this->relu->backward(t4);
    // auto t2 = this->ll1->backward(t3);
    // auto t1 = this->gcn1->backward(t2);
  }

  void update(int learning_rate){
    // this->ll1.update(learning_rate);
    // this->ll2.update(learning_rate);
  }

};
