#pragma once

class SparseMap{

  public:
  SparseMap(int num_nodes){

  }

  int& operator[](long nd);


  void clear();

  ~SparseMap();

}
