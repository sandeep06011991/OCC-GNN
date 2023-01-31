#include "graph/dataset.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <assert.h>
#include <util/cuda_utils.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

Dataset::Dataset(std::string dir, bool testing){
  this->BIN_DIR = dir;
  this->testing = testing;
  read_meta_file();
  read_graph();
  read_node_data();
  // assert(noClasses != 0);
  // read_training_splits();
}

void Dataset::read_graph(){
  // Different file format as sampling needs csc format or indegree graph
  std::fstream file1(this->BIN_DIR + "/cindptr.bin",std::ios::in|std::ios::binary);
  long  * _indptr = (long *)malloc ((this->num_nodes + 1) * sizeof(long));
  gpuErrchk(cudaMalloc((void**)&this->indptr, ((this->num_nodes + 1) * sizeof(long))));
  file1.read((char *)_indptr,(this->num_nodes + 1) * sizeof(long));
  gpuErrchk(cudaMemcpy(this->indptr, _indptr, (this->num_nodes + 1) * sizeof(long) , cudaMemcpyHostToDevice));
  thrust::device_vector<long> temp(this->indptr,this->indptr +  this->num_nodes + 1);
  long sum = thrust::reduce(temp.begin(), temp.end(), 0, thrust::plus<long>());

  std::fstream file2(this->BIN_DIR + "/cindices.bin",std::ios::in|std::ios::binary);
  long * _indices = (long *)malloc ((this->num_edges) * sizeof(long));
  file2.read((char *)_indices,(this->num_edges) * sizeof(long));
  gpuErrchk(cudaMalloc((void**)&this->indices, ((this->num_edges) * sizeof(long))));
  gpuErrchk(cudaMemcpy(this->indices, _indices, (this->num_edges) * sizeof(long) , cudaMemcpyHostToDevice));

  free(_indptr);
  free(_indices);
  // Fixme: ADD corect checksums
  // assert(s ==  csum_edges );
}

void Dataset::read_node_data(){
    std::fstream file2(this->BIN_DIR + "/partition_map_opt.bin",std::ios::in|std::ios::binary);
    int * _partition_map = (int *)malloc (this->num_nodes *  sizeof(int));
    file2.read((char *)_partition_map,this->num_nodes *  sizeof(int));
    gpuErrchk(cudaMalloc((void**)&this->partition_map, (this->num_nodes *  sizeof(int))));
    gpuErrchk(cudaMemcpy(this->partition_map, _partition_map, (this->num_nodes *  sizeof(int)) , cudaMemcpyHostToDevice));
    free(_partition_map);
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
