#include "graph/sample.h"
#include "graph/sliced_sample.h"
#include "transform/slice.h"
#include <iostream>

// Hardcoded Sample 3 Hop
// l1 nodes = 1 2
// l1 edges = (1 2 3 4)(2 3 4)
// l2 nodes = (1 2 3 4)
// l2 edges = (1) (1 2 3)(3 4)(8 4)
// l3 nodes = (1 2 3 4 8)
// l3 edges = (1) (2 3) (3) (4 5) (8 3)
void getDefaultSample(Sample &s){
  Block *b = s.block[0];
  b->layer_nds.insert(b->layer_nds.end(),{1,2});
  b = s.block[1];
  b->offsets.insert(b->offsets.end(),{0,4,7});
  b->in_degree.insert(b->in_degree.end(),{4,3});
  b->indices.insert(b->indices.end(),{1,2,3,4,2,3,4});
  b->layer_nds.insert(b->layer_nds.end(), {1,2,3,4});
  std::cout << b->layer_nds.size() << s.block[1]->layer_nds.size() <<"\n";
  b = s.block[2];
  b->offsets.insert(b->offsets.end(),{0,1,4,6,8});
  b->in_degree.insert(b->in_degree.end(),{1,3,2,2});
  b->indices.insert(b->indices.end(),{1,1,2,3,3,4,2,8,4});
  b->layer_nds.insert(b->layer_nds.end(), {1,2,3,4,8});
  b = s.block[3];
  b->offsets.insert(b->offsets.end(),{0,1,3,4,6,8});
  b->in_degree.insert(b->in_degree.end(),{1,3,2,2,2});
  b->indices.insert(b->indices.end(),{1,2,3,3,4,5,2,8,3});
  b->layer_nds.insert(b->layer_nds.end(), {1,2,3,4,8});
  std::cout << b->layer_nds.size() << s.block[2]->layer_nds.size() <<"\n";

  b = s.block[3];
  b->offsets.insert(b->offsets.end(),{0,1,3,4,6,8});
  b->in_degree.insert(b->in_degree.end(),{1,2,1,2,2});
  b->indices.insert(b->indices.end(),{1,2,3,3,4,5,8,3});
  b->layer_nds.insert(b->layer_nds.end(), {1,2,3,4,5,8});
  std::cout << b->layer_nds.size() << s.block[3]->layer_nds.size() <<"\n";

}

void getHardAnserFullCache(PartitionedSample &s);

void getHardAnserNoCache(PartitionedSample &s);

void test_full_cache_case(){
  // Workload is partitioned %4. Storage is fully replicated.
}
int main(){
  Sample s(3);
  PartitionedSample ps;
  getDefaultSample(s);
  std::vector<int> workload_map{0,1,2,3,0,1,2,3,0};
  std::vector<int> storage_map_full[4];
  std::vector<int> storage_map_nil[4];

  for(int i=0;i<4;i++){
    auto end = storage_map_full[i].end();
    storage_map_full[i].insert(end,{0,1,2,3,4,5,6,7,8,9});

    end = storage_map_nil[i].end();
    storage_map_nil[i].insert(end,{-1,-1,-1,-1,-1,-1,-1,-1,-1});
  }
  Slice * slice = new Slice(workload_map, storage_map_full);
  slice->slice_sample(s,ps);
  ps.clear();
  Slice * slice1 = new Slice(workload_map, storage_map_nil);
  slice1->slice_sample(s,ps);

}
