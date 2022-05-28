#include "util/duplicate.h"
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;

DuplicateRemover::DuplicateRemover(long num_nodes){
    mask = (int *)malloc(sizeof(int) * num_nodes);
    memset(mask,0,sizeof(int) * num_nodes);
    this->used_nodes.clear();
}

void DuplicateRemover::order_and_remove_duplicates(std::vector<long>& nodes){
    int i = this->used_nodes.size();
    // assert(this->used_nodes.size() == 0);
    int j = 0;
    for(long nd1: nodes){
      if(mask[nd1] == 0){
        //fixme: Potential firehazard
        i++;
        nodes[j] = nd1;
        mask[nd1] = i;
        j++;
        this->used_nodes.push_back(nd1);
      }
    }
    nodes.resize(j);
}

void DuplicateRemover::clear(){
  for(long nd1: this->used_nodes){
    mask[nd1] = 0;
  }
  this->used_nodes.clear();
}

void DuplicateRemover::replace(vector<long> &v){
  int failed = 0;
  for(int i=0;i<v.size();i++){
    int t = v[i];
    v[i] = mask[v[i]]-1;
    if(v[i] == -1){
      std::cout << "failed to find " <<  t <<"\n";
      failed ++ ;
    }
  }
  // std::cout << "failed for " << failed << " " << v.size()  <<"\n";
}


DuplicateRemover::~DuplicateRemover(){
    free(mask);
    this->used_nodes.clear();
}
