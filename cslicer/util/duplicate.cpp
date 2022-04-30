#include "util/duplicate.h"
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
using namespace std;

DuplicateRemover::DuplicateRemover(long num_nodes){
    mask = (int *)malloc(sizeof(int) * num_nodes);
    memset(mask,0,sizeof(int) * num_nodes);
    this->used_nodes.clear();
}

void DuplicateRemover::order_and_remove_duplicates(std::vector<long>& nodes){
    int i = 0;
    assert(this->used_nodes.size() == 0);
    for(long nd1: nodes){
      if(mask[nd1] == 0){
        nodes[i] = nd1;
        i++;
        mask[nd1] = i;
        this->used_nodes.push_back(nd1);
      }
    }
    nodes.resize(i);
}

void DuplicateRemover::clear(){
  for(long nd1: this->used_nodes){
    mask[nd1] = 0;
  }
  this->used_nodes.clear();
}

void DuplicateRemover::replace(vector<long> &v){
  for(int i=0;i<v.size();i++){
    v[i] = mask[v[i]]-1;
  }
}


DuplicateRemover::~DuplicateRemover(){
    free(mask);
    this->used_nodes.clear();
}
