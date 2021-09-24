#pragma once

#include "tensor.hh"
#include "linear.hh"

class LinearLayer  {
private:
	const float weights_init_threshold = 0.01;

	Tensor<float> * W;
	Tensor<float> * b;
  Tensor<float> * out;

	Tensor<float> * dW;
	Tensor<float> * db;
  Tensor<float> * in_grad;

  int dim1;
  int dim2;
  int in_dim;
public:
  LinearLayer(int dim1, int dim2, int in_dim);


	Tensor<float>& computeForwardPass(Tensor<float>& in);
	Tensor<float>& computeBackPass(Tensor<float>& dW);

  // void updateWeights(Optimizer &optm);
};
