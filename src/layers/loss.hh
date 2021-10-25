#include "tensor.hh"
#pragma once

class CrossEntropyLoss{
public:
   int N;
   int D;
   Tensor<float> * exp_x = nullptr;
   Tensor<float> * exp_sum = nullptr;
   Tensor<float> * loss = nullptr;
   Tensor<float> * dx = nullptr;

   void compute_exponent(Tensor<float> &in);
   void compute_exponent_sum();
   void compute_loss(Tensor<float> &in,Tensor<int> &labels);


    // Create space for all data structures
    CrossEntropyLoss(){
      
    }
    CrossEntropyLoss(int N, int D);
    // For classification
    // in Shape = N,C
    // N num examples, C num examples.
    // returns Tensor of shape N,1
    Tensor<float>& forward(Tensor<float> &in,Tensor<int> &true_labels);

    Tensor<float>& backward(Tensor<int> &true_labels);

};
