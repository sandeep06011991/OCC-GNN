#pragma once
#include <vector>
#include<bits/stdc++.h>
#include<stdio.h>
// Node ids refer to the ordering in the main graph.
class SampleLayer{

public:
  std::vector<int> nd1;

  std::vector<int> nd2;

  std::vector<std::pair<int,int>> edges;

  std::vector<int> indptr;
  std::vector<int> indices;
  int in_nodes;
  int out_nodes;

  void clear(){
    nd1.clear();
    nd2.clear();
    edges.clear();
    indptr.clear();
    indices.clear();
    in_nodes = 0;
    out_nodes = 0;
  }

  void remove_duplicates(){
    // Assume edges are not repeated.
    std::sort(nd1.begin(), nd1.end()); // {1 1 2 3 4 4 5}
    auto last = std::unique(nd1.begin(), nd1.end());
    nd1.erase(last, nd1.end());
    std::sort(nd2.begin(), nd2.end()); // {1 1 2 3 4 4 5}
    last = std::unique(nd2.begin(), nd2.end());
    nd2.erase(last, nd2.end());
  }

  void create_csr(){
    std::sort(edges.begin(), edges.end(), [](const std::pair<int,int> &a,
                      const std::pair<int,int> &b){
                        if(a.first == b.first) return a.second < b.second;
                        return a.first < b.first;
                      });
    int curr_size = 0;
    int curr_nd = -1;
    for(int i=0;i<edges.size();i++){
      indices.push_back(edges[i].second);
      if(edges[i].first == curr_nd ){
         curr_size ++;
      }else{
        indptr.push_back(i);
        curr_size = 1;
        curr_nd = edges[i].first;
      }
    }
    indptr.push_back(edges.size());
    assert(indptr.size() == nd1.size()+1);

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
