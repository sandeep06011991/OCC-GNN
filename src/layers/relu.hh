#include "tensor.hh"

#pragma once

class Relu{

  Tensor<float> *out = nullptr;
  Tensor<float> *d_out = nullptr;
  // Need this for gradient.
  int dim1;
  int dim2;

public:
  Relu();

  Relu(int dim1, int dim2);

  Tensor<float>& forward(Tensor<float>& X);

  Tensor<float>& backward(Tensor<float>& grad_x);

};
