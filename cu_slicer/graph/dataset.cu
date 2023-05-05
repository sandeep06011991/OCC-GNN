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

Dataset::Dataset(std::string dir, bool testing, int num_partitions, bool random){
  this->BIN_DIR = dir;
  this->testing = testing;
  this->num_partitions = num_partitions;
  this->random = random;
  read_meta_file();
  read_graph();
  read_node_data();
}

void Dataset::read_graph(){
  // Different file format as sampling needs csc format or indegree graph
  std::fstream file1(this->BIN_DIR + "/cindptr.bin",std::ios::in|std::ios::binary);
  long  * _indptr = (long *)malloc ((this->num_nodes + 1) * sizeof(long));
  file1.read((char *)_indptr,(this->num_nodes + 1) * sizeof(long));
  std::vector<long> _t_indptr(_indptr, _indptr + this->num_nodes+ 1);
  indptr_d = (* new device_vector<long>(_t_indptr));

  long sum = cuslicer::transform::reduce(indptr_d);

  std::fstream file2(this->BIN_DIR + "/cindices.bin",std::ios::in|std::ios::binary);
  long * _indices = (long *)malloc ((this->num_edges) * sizeof(long));
  file2.read((char *)_indices,(this->num_edges) * sizeof(long));
  std::vector<long> _t_indices(_indices, _indices + this->num_edges);
  indices_d = ( * new device_vector<long>(_t_indices));

  free(_indptr);
  free(_indices);
  // Fixme: ADD corect checksums
  // assert(s ==  csum_edges );
}

void Dataset::read_node_data(){
    // Add feature for flexible partition
    // Make shared pointernaj

    int * _partition_map = (int *)malloc (this->num_nodes *  sizeof(int));
    int n_gpu = this->num_partitions;
    if (! this->random){
	std::cout << "read partition " << n_gpu << "\n";
    	std::fstream file2(this->BIN_DIR + "/partition_map_opt_" + std::to_string(this->num_partitions) +".bin",std::ios::in|std::ios::binary);
    	file2.read((char *)_partition_map,this->num_nodes *  sizeof(int));
      n_gpu = this->num_partitions;
    }else{
	    std::cout << "reading random map \n" ;
	assert(this->num_partitions == 4);
	    std::fstream file2(this->BIN_DIR + "/partition_map_opt_random.bin", std::ios::in|std::ios::binary);
    	file2.read((char *)_partition_map,this->num_nodes *  sizeof(int));
    }
    std::vector<int> _t_partition_map(_partition_map, _partition_map + this->num_nodes);
    for(int i : _t_partition_map){
      assert(i < n_gpu);
    }
    partition_map_d = (* new device_vector<int>(_t_partition_map));
    // gpuErrchk(cudaMalloc((void**)&this->partition_map, (this->num_nodes *  sizeof(int))));
    // gpuErrchk(cudaMemcpy(this->partition_map, _partition_map, (this->num_nodes *  sizeof(int)) , cudaMemcpyHostToDevice));
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
