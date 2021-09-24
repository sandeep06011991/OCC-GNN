#include "tensor.hh"
#pragma once

class CrossEntropyLoss{
public:
   int N;
   int D;
   Tensor<float> * exp_x;
   Tensor<float> * exp_sum;
   Tensor<float> * loss;
   Tensor<float> * dx;

   void compute_exponent(Tensor<float> &in);
   void compute_exponent_sum();
   void compute_loss(Tensor<float> &in,Tensor<int> &labels);


    // Create space for all data structures
    CrossEntropyLoss(int N, int D);
    // For classification
    // in Shape = N,C
    // N num examples, C num examples.
    // returns Tensor of shape N,1
    Tensor<float>& lossForward(Tensor<float> &in,Tensor<int> &true_labels);

    Tensor<float>& lossBackward(Tensor<int> &true_labels);

};
