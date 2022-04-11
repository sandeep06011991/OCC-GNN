#include<iostream>
#include<string>
#include "dataset.h"
#include "slicer.h"
#include<vector>
int main(){
  std::string filename = "ogbn-arxiv";
  std::string DATA_DIR = "/data/sandeep/";
  Dataset *dataset = new Dataset(DATA_DIR + filename);

  std::vector<int> storage_map[4];
  Slicer *slicer = new Slicer(dataset, storage_map, 4096);
  std::cout <<"hello world\n";
}
