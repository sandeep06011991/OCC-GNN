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
  // len indptr in range of nd1

  // Local Edge Ids
  std::vector<int> indices;
  // indices in range of nd2

  std::unordered_map<int,int> reorder_map;

  int in_nodes;
  // in_nodes = size(nd2)
  int out_nodes;
  // out_nodes = size(nd1)

  void clear(){
    nd1.clear();
    nd2.clear();
    edges.clear();
    indptr.clear();
    indices.clear();
    reorder_map.clear();
    in_nodes = 0;
    out_nodes = 0;
  }

  void check_consistency(){
     assert(indptr.size() == out_nodes + 1);
     for(int i=0;i<indices.size();i++){
       assert(indices.data()[i] < in_nodes);
     }
  }

  void remove_duplicates(){
    // Assume edges are not repeated.
    std::sort(nd1.begin(), nd1.end()); // {1 1 2 3 4 4 5}
    auto last = std::unique(nd1.begin(), nd1.end());
    nd1.erase(last, nd1.end());
    std::sort(nd2.begin(), nd2.end()); // {1 1 2 3 4 4 5}
    last = std::unique(nd2.begin(), nd2.end());
    nd2.erase(last, nd2.end());
    in_nodes = nd2.size();
    out_nodes = nd1.size();
    for(int i=0;i<in_nodes;i++){
      reorder_map[nd2.data()[i]] = i;
    }

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
      indices.push_back(reorder_map[edges[i].second]);
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
    check_consistency();
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
