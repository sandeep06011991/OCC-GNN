#include <torch/torch.h>

__global__ 
void kernel(torch::Tensor at){
    printf("kernel %x\n", at.data_ptr());
}

int main(){
  auto opts = torch::TensorOptions().dtype(torch::kInt32)\
      .device(torch::kCUDA, 0);

  auto data = torch::empty({(signed long) 10,}, opts );
  std::cout << data.data_ptr() << " is the pointer \n";
  kernel<<<1,1>>>(data);
  std::cout << "done\n";
}