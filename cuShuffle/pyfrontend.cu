#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include<time.h>

namespace py = pybind11;

torch::Tensor get_tensor(){
  // std::vector<data>
  auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 1)
    .requires_grad(true);

  torch::Tensor tensor = torch::randint(/*high=*/10, {5, 5}, options);
  cudaSetDevice(2);
  float *a;
  cudaMalloc(&a, sizeof(a) * 25);
  cudaMemcpy(a, (void *)tensor.data_ptr<float>(), sizeof(float) * 25 , cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  return tensor;
}

void copy_tensor(torch::Tensor a, torch::Tensor b){
  int src = a.device().index();
  cudaSetDevice(src);
  auto sizes = a.sizes();
  int p = 1;
  for(auto s:sizes)p *= s;
  auto x = (void *)a.data_ptr();
  auto y = (void *)b.data_ptr();
  for(int k=0;k<4;k++){
    clock_t begin_time = clock();
    cudaMemcpy(x, y, sizeof(float) * p, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    float t =  float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    std::cout << "Move" <<( p * 4)/(1024 * 1024 * 1024 ) <<"GB Bandwidth" << 1/t << "GBps\n";
  }
}

void test_bandwidth(){
  float *a, *b;
  size_t size = (1024 * 1024 * 1024);
  cudaSetDevice(0);
  cudaMalloc(&a, size);
  cudaSetDevice(1);
  cudaMalloc(&b, size);
  for(int i=0;i<10;i++){
    clock_t begin_time = clock();
    cudaSetDevice(0);
    cudaMemcpy(a,b, size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    float t =  float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    std::cout << "Bandwidth" << 1/t << "GBps\n";
  }
  cudaSetDevice(0);
  cudaFree(a);
  cudaSetDevice(1);
  cudaFree(b);
}
PYBIND11_MODULE(shuffle, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("get_tensor", &get_tensor);
    m.def("copy_tensor", &copy_tensor);
    m.def("test_bandwidth",&test_bandwidth);
}
