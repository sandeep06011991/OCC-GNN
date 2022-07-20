#include <torch/extension.h>

#include <iostream>
#include <vector>
using namespace std;
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

torch::Tensor getlist(){
	vector<int> *v = new vector<int>();
	for(int i=0;i<10000 * 100; i++){
		v->push_back(i);
	}
	auto opts = torch::TensorOptions().dtype(torch::kInt);
	torch::Tensor t = torch::from_blob(v->data(), {10000 * 100}, opts);
	return t;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid", &d_sigmoid, "Forward");
  m.def("getlist", &getlist, "get list"); 
}
