#include <pybind11/pybind11.h>
#include <torch/extension.h>


namespace py = pybind11;

torch::Tensor add_two_tensors(torch::Tensor a, torch::Tensor b){
	auto options =  torch::TensorOptions()\
     .dtype(torch::kFloat32)\
    .layout(torch::kStrided)\
    .device(torch::kCUDA, 1)\
    .requires_grad(false);
	torch::Tensor tensor = torch::full((2,3),5,options);
	return tensor;

}


PYBIND11_MODULE(cuslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add_two_tensors", &add_two_tensors);
}
