#pragma once

// Currently a matrix(i.e 2D),
// extend later for more complex shapes.
template<typename T>
class Tensor {

private:
	bool device_allocated;
	bool host_allocated;
  int dim1;
  int dim2;

	void allocateCudaMemory();
	void allocateHostMemory();

public:

	T* data_device;
	T* data_host;

	Tensor(size_t x_dim = 1, size_t y_dim = 1);
  Tensor(T* data, int  x_dim, int y_dim);

	void allocateMemory();

	void copyHostToDevice();
	void copyDeviceToHost();

	// float& operator[](const int index);
	// const float& operator[](const int index) const;
};

template class Tensor<float>;
template class Tensor<int>;
