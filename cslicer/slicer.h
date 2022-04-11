#include "dataset.h"
#pragma once
#include <vector>
class Slicer{

private:

  Dataset *dataset;
  std::vector<int> storage_map[4];
  int batch_size;

public:
  Slicer(Dataset * dataset, std::vector<int> storage_map[4], int batch_size){
      this->dataset = dataset;
      for(int i=0;i<4;i++){
        this->storage_map[i] = storage_map[i];
      }
      this->batch_size = batch_size;
  }

  int getNumberOfBatches();

  void shuffle();

};
