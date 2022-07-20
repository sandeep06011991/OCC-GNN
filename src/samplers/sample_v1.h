#pragma once
#include <vector>
#include<bits/stdc++.h>
#include<stdio.h>


// Node ids refer to the ordering in the main graph.
class SampleLayer{

public:
  std::vector<long> nd1;

  std::vector<long> nd2;

  std::vector<long> indptr;
  // len indptr in range of nd1

  std::vector<long> indices;
  // indices in range of nd2

  int in_nodes;
  // in_nodes = size(nd2)
  int out_nodes;
  // out_nodes = size(nd1)


  void clear(){
    nd1.clear();
    nd2.clear();
    indptr.clear();
    indices.clear();
    in_nodes = 0;
    out_nodes = 0;
  }

  void check_consistency(){
     assert(indptr.size() == out_nodes + 1);
     for(int i=0;i<indices.size();i++){
       assert(indices.data()[i] < in_nodes);
     }
  }

  void reorder(int * reorder_map){
    int current = 1;
    this->nd2.clear();
    for(int i = 0;i<this->indices.size();i++){
      int nd2 = this->indices[i];
      if(reorder_map[nd2]==0){
        reorder_map[nd2] = current;
        this->nd2.push_back(nd2);
        current ++;
      }
      this->indices[i] = reorder_map[nd2]-1;
    }
    // clean up
    for(int i=0;i<this->nd2.size();i++){
      reorder_map[this->nd2[i]] = 0;
    }
    in_nodes = nd2.size();
    out_nodes = nd1.size();
  }

};

class TwoHopSample{

public:
  SampleLayer l1;
  SampleLayer l2;

  void clear(){
    l1.clear();
    l2.clear();
  }
};


class ThreeHopSample{
public:
  SampleLayer l1;
  SampleLayer l2;
  SampleLayer l3;

  void clear(){
    l1.clear();
    l2.clear();
    l3.clear();
  }
};


class KHopSample{
  static const int MAX_LAYERS = 10;
public:
  SampleLayer ll[MAX_LAYERS];
  int k_hop;
  KHopSample(int k_hop){
      assert(k_hop < MAX_LAYERS);
      this->k_hop = k_hop;
  }

  void clear(){
      for(int i=0;i<k_hop;i++){
        this->ll[i].clear();
      }
  }
};
