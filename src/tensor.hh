#pragma once

// Currently a matrix(i.e 2D),
// extend later for more complex shapes.
template<typename T>
class Tensor {

private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:

	T* data_device;
	T* data_host = nullptr;
  int dim1;
  int dim2;


	Tensor(int x_dim = 1,int y_dim = 1);
  Tensor(T* data, int  x_dim, int y_dim);

	void allocateMemory();

	void copyHostToDevice();
	void copyDeviceToHost();

	// float& operator[](const int index);
	// const float& operator[](const int index) const;
  void verify(T* correct_data);
  void debugTensor();
};

template class Tensor<float>;
template class Tensor<int>;

Tensor<float> * allocate_ones(int dim1, int dim2);

float * allocate_random(int size);

bool tensor_equal(Tensor<float> &a, Tensor<float> &b);

void mat_mul_a_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);

void mat_mul_a_b_t(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);

void mat_mul_a_t_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);
