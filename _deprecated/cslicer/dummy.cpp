#include <torch/extension.h>


int test(){
	auto opts = torch::TensorOptions().dtype(torch::kInt64)
               .device(torch::kCPU);
    // this->gpu_id = bp->gpu_id;
    // num_in_nodes_local = bp->num_in_nodes_local;
    // num_in_nodes_pulled = bp->num_in_nodes_pulled;
    // num_out_local = bp->num_out_local;
    // num_out_remote = bp->num_out_remote;
		int *a;
    auto indptr_L = torch::from_blob(a, {10}, opts);

}

PYBIND11_MODULE(cslicer, m){ 
/* Binding code */
m.def("get_data", &test);

}
