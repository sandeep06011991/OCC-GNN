#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <assert.h>

#include <cub/cub.cuh>
#include "../util/cub.h"
#include "../util/cuda_utils.h"
#include "../graph/dataset.cuh"
#include "../util/device_vector.h" 

using namespace cuslicer;

Dataset::Dataset(std::string dir, int num_partitions, bool random, bool UVA){
  this->BIN_DIR = dir;
  this->num_partitions = num_partitions;
  this->random = random;
  this->UVA = UVA;
  read_meta_file();
  read_graph();
}

void Dataset::read_graph(){
  // Different file format as sampling needs csc format or indegree graph
  std::fstream file1(this->BIN_DIR + "/cindptr.bin",std::ios::in|std::ios::binary);
  if (UVA){
    gpuErrchk(cudaHostAlloc(&indptr_h,(this->num_nodes + 1) * sizeof(NDTYPE), cudaHostAllocMapped | cudaHostAllocWriteCombined ));
    file1.read((char *)indptr_h,(this->num_nodes + 1) * sizeof(NDTYPE));
    gpuErrchk(cudaHostGetDevicePointer(&indptr_d, indptr_h, 0));
  }else{
    NDTYPE  * _indptr = (NDTYPE *)malloc ((this->num_nodes + 1) * sizeof(NDTYPE));
    file1.read((char *)_indptr,(this->num_nodes + 1) * sizeof(NDTYPE));
    gpuErrchk(cudaMalloc(&indptr_d, (this->num_nodes + 1)* sizeof(NDTYPE)));
    gpuErrchk(cudaMemcpy(indptr_d, _indptr,(this->num_nodes + 1)* sizeof(NDTYPE), cudaMemcpyHostToDevice )); 
    free(_indptr);    
  }

  NDTYPE sum = cuslicer::transform<NDTYPE>::reduce_d(indptr_d, this->num_nodes+1);
  std::cout << "Read sum " << sum <<"\n";
    
  std::fstream file2(this->BIN_DIR + "/cindices.bin",std::ios::in|std::ios::binary);
  if (UVA){
    std::cout << "using host alloc\n";
    gpuErrchk(cudaHostAlloc(&indices_h,(this->num_edges) * sizeof(NDTYPE), cudaHostAllocMapped | cudaHostAllocWriteCombined  ));
    file2.read((char *)indices_h,(this->num_edges) * sizeof(NDTYPE));
    gpuErrchk(cudaHostGetDevicePointer(&indices_d, indices_h, 0));
  }else{
    std::cout << "using device alloc \n";
    NDTYPE * _indices = (NDTYPE *)malloc ((this->num_edges) * sizeof(NDTYPE));
    file2.read((char *)_indices,(this->num_edges) * sizeof(NDTYPE));
    gpuErrchk(cudaMalloc(&indices_d, (this->num_edges)* sizeof(NDTYPE)));
    gpuErrchk(cudaMemcpy(indices_d, _indices,(this->num_edges)* sizeof(NDTYPE), cudaMemcpyHostToDevice )); 
    gpuErrchk(cudaMemcpy(_indices, indices_d, 10 * sizeof(NDTYPE), cudaMemcpyDeviceToHost));
    free(_indices);
  };

  // Fixme: ADD corect checksums
  // assert(s ==  csum_edges );
}

void Dataset::read_meta_file(){
  std::fstream file(this->BIN_DIR + "/meta.txt",std::ios::in);
  std::string line;
  while(getline(file,line)){
    std::cout << line <<"\n";
    if(file.eof())break;
    std::string name = line.substr(0,line.find("="));
    std::string token = line.substr(line.find("=") + 1,line.length() );

    long val = stoll(token);
    if (name == "num_nodes") {
      this->num_nodes = val;
      continue;
    }
    if (name == "num_edges") {
      this->num_edges = val;
      continue;
    }
    if (name == "feature_dim") {
      this->fsize = val;
      continue;
    }
    if (name == "csum_features") {
      this->csum_features = val;
      continue;
    }
    if (name == "csum_labels") {
      this->csum_labels = val;
      continue;
    }
    if (name == "csum_offsets") {
      this->csum_offsets = val;
      continue;
    }
    if (name == "csum_edges") {
      this->csum_edges = val;
      continue;
    }
    if (name == "num_classes") {
      this->noClasses = val;
      continue;
    }
  }
}
