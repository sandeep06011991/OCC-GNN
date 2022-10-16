#include "graph/dataset.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <assert.h>
#include <spdlog/spdlog.h>


Dataset::Dataset(std::string dir, bool testing){
  this->BIN_DIR = dir;
  this->testing = testing;
  read_meta_file();
  read_graph();
  read_node_data();
  spdlog::info("Read graph with num nodes {}",(num_nodes));

  // assert(noClasses != 0);
  // read_training_splits();
}

void Dataset::read_graph(){
  // Different file format as sampling needs csc format or indegree graph
  std::fstream file1(this->BIN_DIR + "/cindptr.bin",std::ios::in|std::ios::binary);
  this->indptr = (long *)malloc ((this->num_nodes + 1) * sizeof(long));
  file1.read((char *)this->indptr,(this->num_nodes + 1) * sizeof(long));
  long s = 0;
  for(long i=0;i<this->num_nodes + 1;i++){
    s = s + this->indptr[i];
  }
  std::cout << "Checksum of indptr and indices skipped \n";
  std::cout << "indptr" << s << " " << csum_offsets << "\n";
  // Fixme: ADD correct checksums
  // assert(s == csum_offsets );
  std::fstream file2(this->BIN_DIR + "/cindices.bin",std::ios::in|std::ios::binary);
  this->indices = (long *)malloc ((this->num_edges) * sizeof(long));
  file2.read((char *)this->indices,(this->num_edges) * sizeof(long));
  s = 0;
  for(long i=0;i<this->num_edges;i++){
    s = s + this->indices[i];
  }
  // Fixme: ADD corect checksums
  // assert(s ==  csum_edges );
}

void Dataset::read_node_data(){
  // string fileName = "test.bin";
  // Skip features not needed.
  // std::fstream file(this->BIN_DIR + "/features.bin",std::ios::in|std::ios::binary);
  // this->features = (float *)malloc (this->num_nodes * this->fsize * sizeof(float));
  // file.read((char *)this->features,this->num_nodes * this->fsize * sizeof(float));
  float s = 0;
  // for(long i=0;i< (this->fsize*this->num_nodes) ;i++){
  //   s = s + this->features[i];
  // }
  // std::cout << "features:" << s << " " << this->csum_features <<"\n";
  // assert(abs(s-this->csum_features)<100);

  // std::fstream file1(this->BIN_DIR + "/labels.bin",std::ios::in|std::ios::binary);
  // this->labels = (int *)malloc (this->num_nodes *  sizeof(int));
  // file1.read((char *)this->labels,this->num_nodes *  sizeof(int));
  // s = 0;
  // for(int i=0;i< (this->num_nodes) ;i++){
  //   s = s + this->labels[i];
  // }
  // assert(s-this->csum_labels<10);
  if (this->testing){
    this->partition_map = (int *)malloc (this->num_nodes *  sizeof(int));
    for(int i=0;i< (this->num_nodes) ;i++){
       this->partition_map[i]= 0;
    }
    return;
  }
  std::fstream file2(this->BIN_DIR + "/partition_map_opt.bin",std::ios::in|std::ios::binary);
  this->partition_map = (int *)malloc (this->num_nodes *  sizeof(int));
  file2.read((char *)this->partition_map,this->num_nodes *  sizeof(int));
  s = 0;
  for(int i=0;i< (this->num_nodes) ;i++){
     this->partition_map[i] < 4;
     s = s + this->partition_map[i];
  }
  assert(s>10);
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
