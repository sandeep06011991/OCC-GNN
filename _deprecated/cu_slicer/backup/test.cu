#include <pybind11/pybind11.h>
#include <torch/extension.h>


namespace py = pybind11;

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int check() {
  // Generate 32M random numbers serially.
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  // Transfer data to the device.
  thrust::device_vector<int> d_vec = h_vec;

  // Sort data on the device.
  thrust::sort(d_vec.begin(), d_vec.end());

  // Transfer data back to host.
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}

torch::Tensor add_two_tensors(torch::Tensor a, torch::Tensor b){
	auto options =  torch::TensorOptions()\
     .dtype(torch::kFloat32)\
    .layout(torch::kStrided)\
    .device(torch::kCUDA, 1)\
    .requires_grad(false);
	torch::Tensor tensor = torch::full((2,3),5,options);
	check();
	return tensor;

}

__global__ 
void get_indptr_required(long *indptr, long *indices,\
	       		long *seeds, long *sz, int size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size){
			sz[tid] = indptr[seeds[tid] + 1] - indptr[seeds[tid]];
			// printf("%ld %d\n",sz[tid], tid);  
	}
}

__global__
void sample(long *indptr, long *indices, long *seeds, long *s_offsets, long *s_indices, int num_seeds){
	int nodeId = blockIdx.x;
	long node = seeds[nodeId];
	int nbId = threadIdx.x;
	if(nodeId < num_seeds){
		int nbs = indptr[node+1] - indptr[node];
		int sample_offset_start = s_offsets[nodeId];
		int node_offsets_start = indptr[node];
		while(nbId < nbs){
			s_indices[sample_offset_start + nbId] = indices[node_offsets_start + nbId];
			nbId +=  blockDim.x;
		}	
	}

}

class PyDummy{
  public:
   torch::Tensor a;
   torch::Tensor b;
   torch::Tensor c;
   PyDummy(){}
};


PyDummy sample_layer(torch::Tensor indptr, torch::Tensor indices, \
		torch::Tensor seeds){
	int num_seeds = seeds.sizes()[0];
	cudaSetDevice(1);
	//std::cout << "num seeds" << num_seeds <<"\n";
	auto options =  torch::TensorOptions()\
	     .dtype(torch::kInt64)\
    	     .layout(torch::kStrided)\
    	     .device(torch::kCUDA, 1)\
    	     .requires_grad(false);
	int num_blocks = (num_seeds - 1)/32 + 1;
	thrust::device_vector<long> s_indptr(num_seeds + 1);
	//torch::Tensor indptr = torch::full({num_seeds},0, options);	
	//torch::Tensor out = torch::full({num_seeds}, 0, options);
	get_indptr_required<<<num_blocks,32>>>((long *)indptr.data_ptr(),(long *)indices.data_ptr(),\
		       		(long *)seeds.data_ptr(), (long *)s_indptr.data().get(), (num_seeds)); 
	cudaDeviceSynchronize();
	long sv = s_indptr[num_seeds - 1];
	thrust::exclusive_scan(s_indptr.begin(), s_indptr.end(), s_indptr.begin());
	cudaDeviceSynchronize();
	s_indptr[num_seeds] = s_indptr[num_seeds-1] + sv;
	//std::cout << s_indptr[num_seeds] <<"\n";
	torch::Tensor s_indices = torch::full({s_indptr[num_seeds]},0, options);
	sample<<<num_seeds, 32>>>((long *) indptr.data_ptr(), (long *) indices.data_ptr(),\
			(long *)seeds.data_ptr(), (long *)(s_indptr.data().get()), \
		       	(long *)s_indices.data_ptr(),\
			num_seeds);
	cudaDeviceSynchronize();
	torch::Tensor s_indptr_v = torch::full({num_seeds + 1},0,options);
	gpuErrchk(cudaMemcpy(s_indptr_v.data_ptr(), s_indptr.data().get(), (num_seeds + 1)* sizeof(long) , cudaMemcpyDeviceToDevice));

	thrust::device_vector<long> d_vec(s_indptr[num_seeds]);
        thrust::copy(d_vec.begin(), d_vec.end(), (long *)s_indices.data_ptr());
	thrust::sort(d_vec.begin(), d_vec.end());
	d_vec.erase(thrust::unique(d_vec.begin(), d_vec.end()), d_vec.end());

	torch::Tensor s_nodes = torch::full({(long)d_vec.size()},0, options);
	cudaMemcpy(s_nodes.data_ptr(), d_vec.data().get(), (d_vec.size()* sizeof(long)) , cudaMemcpyDeviceToDevice);
		

	PyDummy py;
	py.a = s_indptr_v;
	py.b = s_indices;
	py.c = s_nodes;
	return std::move(py);

}

// Sort vector and return vector. 
void thrust_test(torch::Tensor test){
	cudaSetDevice(1);
	int sz = test.sizes()[0];
	thrust::device_vector<long> d_vec((long *)test.data_ptr(), (long *)test.data_ptr() + sz);
	thrust::sort(d_vec.begin(), d_vec.end());
	thrust::copy(d_vec.begin(), d_vec.end(), (long *)test.data_ptr());
}



PyDummy test_object_movement(){
	 auto options =  torch::TensorOptions()\
             .dtype(torch::kInt64)\
             .layout(torch::kStrided)\
             .device(torch::kCUDA, 1)\
             .requires_grad(false);
        PyDummy obj;
	 torch::Tensor a = torch::full({100},12, options);
	 torch::Tensor b = torch::full({100}, 13, options);
	 std::cout << a[0];
	 torch::Tensor c= torch::cumsum(a,0);
	 obj.a = c;
	 obj.b = b;
	 return std::move(obj);
}


PYBIND11_MODULE(cuslicer, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add_two_tensors", &add_two_tensors);
    m.def("sample_layer", &sample_layer);
    m.def("thrust_test", &thrust_test);
    m.def("test_object_movement",&test_object_movement);
    py::class_<PyDummy>(m,"PyDummy")
       .def_readwrite("a",&PyDummy::a)
       .def_readwrite("b", &PyDummy::b)
       .def_readwrite("c",&PyDummy::c);
}
