#include <iostream>
#include <torch/torch.h>
#include "util/torch_device_vector.h"

int main(){
	std::vector<int> test = {1,3,4,5};
	std::cout << "Hello world\n";
	 torch::Tensor tensor = torch::eye(3);
	 device_vector<int> a(test);
	 a.debug();
	std::cout << tensor << std::endl;
}
