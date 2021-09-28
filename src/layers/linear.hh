#pragma once

#include "tensor.hh"
#include "linear.hh"

class LinearLayer  {
public:
	const float weights_init_threshold = 0.01;

  Tensor<float> * in;
	Tensor<float> * W;
	Tensor<float> * b;
  Tensor<float> * out;

	Tensor<float> * dW;
	Tensor<float> * db;
  Tensor<float> * out_grad;
  Tensor<float> * _btemp;

  int dim1;
  int dim2;
  int in_dim;

  LinearLayer(int dim1, int dim2, int in_dim);
// For debugging.
  LinearLayer(float *W, float *B, int dim1, int dim2, int in_dim);

	Tensor<float>& computeForwardPass(Tensor<float> * in);
	Tensor<float>& computeBackwardPass(Tensor<float>& dW);

  // void updateWeights(Optimizer &optm);
};
