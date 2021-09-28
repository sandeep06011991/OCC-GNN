#include "full.h"
// Clean up pseudo code
// completely

class SubGraph{

  // parent Graph.
  Graph * graph;
  vector<int> src_ids;
  vector<int> offset;
  // len(src_ids) == len(offsets)
  vector<int> dest_indices;
  vector<int> dest_ids;
  // len(reordered_ids) == len(dest_indices)
  //  reordered_ids \in [0,len(dest_ids)]
  vector<int> reordered_ids;

  SubGraph();

  void clear();

  void reorder(){
    int *ndArray = malloc(sizeof(noNodes) * int);
    count = 0;
    for(int i=0;i<dest_ids.length;i++){
      ndArray[dest_ids[i]] = count;
      count ++;
    }
  }

}

float * collectNodeFeatures(float * nodeFeatures, int fsize, int trueIds){
    float * out;
    for(int i=0;i<noTrueIds;i++){
      for(int j=0;j<fsize;j++){
          out[i*fsize+j] = nodeFeatures[trueIds[i] * fsize + j]
      }
    }
    return out;
}


TwoHopNoSample::sample_from_hop(Sample *S,int *target_vertices, int no_targets){
  // K-Hop Sampling creates 2 subgraphs;
  // Only related to graph Structure.
  prefix_sum = 0;
  for(int i=0;i<no_targets;i++){
    int nodeId = target_vertices[i];
    S1.offsets.push_back(prefix_sum);
    S1.src_ids.push_back(nodeId);
    for(int j = offsets[nodeId];j<offsets[nodeId+1];i++){
      S1.dest_indices.push_back(indices[j])
    }
    prefix_sum = prefix_sum + (offsets[nodeId+1] - offsets[nodeId]);
  }
  dest_ids.clear();
  dest_ids.push_back(S1.dest_indices);
  std::sort(dest_ids.begin(), dest_ids.end()); // {1 1 2 3 4 4 5}
  auto last = std::unique(dest_ids.begin(), dest_ids.end());
  dest_ids.erase(last, dest_ids.end());
}

void TwoHopNoSample::sample_from_target(int * target_vertices, int no_targets){


  this->sample_from_hop(&this->l1,target_vertices, no_targets);
  this->sample_from_hop(&this->l2,S1->dest_ids.data, S1->dest_ids.size());

  // For each neighbourhood sample, fill up reordering vectors.

  s1.reorder();
  s2.reorder();
  // slice neighbourhoods.

}
