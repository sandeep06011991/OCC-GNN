

class NeighbourSampler{
	torch::Tensor indptr;
	torch::Tensor indices;
	public:
	NeibourSampler(torch::Tensor indptr, torch::Tensor indices){
  		this->indptr = indptr;
		this->indices = indices;
       	}


	void sample_layers(torch::Tensor seeds){
		cudaSetDevice(1);
		thrust::device_vector<long> indptr(seeds_size);
		//get offsets 
		thurst::device_vector<long> indices(indptr[size]-1);
		// Sample and fill up indices ptr
		// Convert everything in block

	}

}	
