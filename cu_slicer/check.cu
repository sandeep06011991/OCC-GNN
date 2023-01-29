
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <torch/extension.h>
#include <thrust/device_vector.h>

using namespace std;
namespace py = pybind11;


torch::Tensor getTensor(){
	cudaSetDevice(0);
	thrust::host_vector<int> v;
	v.resize(19);
	for(int i=0;i < 19; i++){
		v[i] = i;
	}
	thrust::device_vector<int> v1 = v;
	torch::Tensor t;
	auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
	
	t = torch::from_blob((int *)thrust::raw_pointer_cast(v1.data()), {19}, opts).clone();
	return std::move(t);	
}

PYBIND11_MODULE(cuslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("getCUDATensor", getTensor, py::return_value_policy::take_ownership);
}
