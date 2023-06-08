
#include <pybind11/pybind11.h>

namespace py = pybind11;

torch::Tensor add_two_tensors(torch::Tensor a, torch::Tensor b){
	auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 1)
    .requires_gra(false);
	torch::Tensor tensor = torch::fill((2,3),5,options);
	return tensor;

}


PYBIND11_MODULE(cslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add_two_tensors", &add_two_tensors);
}
