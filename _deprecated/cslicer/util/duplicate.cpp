#include "duplicate.h"
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;


ArrayMap::ArrayMap(long num_nodes){
    mask = (int *)malloc(sizeof(int) * num_nodes);
    memset(mask,0,sizeof(int) * num_nodes);
    this->used_nodes.clear();
}



void ArrayMap::order_and_remove_duplicates(std::vector<long>& nodes){
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

void ArrayMap::order(std::vector<long> &nodes){
  int i = this->used_nodes.size();
  for(long nd1: nodes){
      if(mask[nd1] == 0){
        i++;
	mask[nd1] = i;
        this->used_nodes.push_back(nd1);
      }
    }

}

void ArrayMap::clear(){
  for(long nd1: this->used_nodes){
    mask[nd1] = 0;
  }
  this->used_nodes.clear();
}

void ArrayMap::replace(vector<long> &v){
  int failed = 0;
  for(int i=0;i<v.size();i++){
    long t = v[i];
    v[i] = mask[v[i]]-1;
    if(v[i] == -1){
      std::cout << "failed to find " <<  t <<"\n";
      failed ++ ;

      assert(false);
    }
  }
  // std::cout << "failed for " << failed << " " << v.size()  <<"\n";
}

ArrayMap::~ArrayMap(){
    free(mask);
    this->used_nodes.clear();
}

HashMap::HashMap(long num_nodes){
    map.clear();
    count = 0;
}

void HashMap::clear(){
  map.clear();
  count = 0;
}

void HashMap::order_and_remove_duplicates(std::vector<long>& nodes){
    // assert(this->used_nodes.size() == 0);
    int j = 0;
    for(long nd1: nodes){
      auto search = map.find(nd1);
      if(search == map.end()){
        //fixme: Potential firehazard
        count++;
        map.insert(std::make_pair(nd1, count));
        nodes[j] = nd1;
        j++;
      }
    }
    nodes.resize(j);
}

void HashMap::order(std::vector<long> &nodes){
  for(long nd1: nodes){
    auto search = map.find(nd1);
    if(search == map.end()){
      //fixme: Potential firehazard
      count++;
      map.insert(std::make_pair(nd1, count));
    }
  }

}

void HashMap::replace(vector<long> &v){
  int failed = 0;
  for(int i=0;i<v.size();i++){
    long t = v[i];
    auto search = map.find(t);

    if(search == map.end()){
      std::cout << "failed to find " <<  t <<"\n";
      failed ++ ;
      assert(false);
    }
    v[i] = search->second -1;
  }
  // std::cout << "failed for " << failed << " " << v.size()  <<"\n";
}
