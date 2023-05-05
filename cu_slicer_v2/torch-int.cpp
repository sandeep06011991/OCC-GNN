#include <iostream>
#include <torch/torch.h>

int main(){
	std::vector<int> test = {1,3,4,5};
	std::cout << "Hello world\n";
	 torch::Tensor tensor = torch::eye(3);
	  std::cout << tensor << std::endl;
}
