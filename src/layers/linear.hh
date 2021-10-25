#pragma once

#include "tensor.hh"
#include "linear.hh"

class LinearLayer  {

  Tensor<float> in;

public:
	const float weights_init_threshold = 0.01;

  Tensor<float> * W = nullptr;
	Tensor<float> * b = nullptr;
  Tensor<float> * out = nullptr;

	Tensor<float> * dW;
	Tensor<float> * db;
  Tensor<float> * out_grad= nullptr;
  Tensor<float> * _btemp= nullptr;

  int dim1;
  int dim2;
  int in_dim;

  LinearLayer(int dim1, int dim2);

  LinearLayer(int dim1, int dim2, int in_dim);

// For debugging.
  LinearLayer(float *W, float *B, int dim1, int dim2, int in_dim);
  LinearLayer(float *W, float *B,int dim1, int dim2);

	Tensor<float>& forward(Tensor<float>& in);
	Tensor<float>& backward(Tensor<float>& dW);

  void update(float learning_rate);
};
