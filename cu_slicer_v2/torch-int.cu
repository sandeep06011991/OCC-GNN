#include <iostream>
#include <torch/torch.h>
#include "util/device_vector.h"

int main(){
	cudaSetDevice(0);
	std::vector<int> test = {1,3,4,5};
	// torch::Tensor tensor = torch::eye(3);
	// auto opts = torch::TensorOptions().dtype(torch::kInt32)\
    //   .device(torch::kCUDA, 0);
	// size_t sz = 10;
    // auto data = torch::ones({(signed int)sz,(signed int)sz,}, opts );
	// opts = torch::TensorOptions().dtype(torch::kInt32)\
    //   .device(torch::kCUDA, 0); 
    // auto tdata = torch::empty({(signed int)test.size(),}, opts );
	
    std::cout << "created works outisdee\n";
	cuslicer::device_vector<int> a(test);
	 a.debug("print me");
	 std::cout << "check " <<  test[3] << "\n";
	//  std::cout << data << std::endl;
}
