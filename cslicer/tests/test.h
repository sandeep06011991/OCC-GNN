#pragma once
#include<vector>
#include "graph/sample.h"
#include "graph/sliced_sample.h"




void test_sample_partition_consistency(Sample &s,
        PartitionedSample &p,
      std::vector<int> storage_map[8], int gpu_capacity[8], int num_nodes, int num_gpus);


void test_sample_partition_consistency_gat(Sample &s,
              PartitionedSample &p,
            std::vector<int> storage_map[4], int gpu_capacity[4], int num_nodes);

void  test_pull_benefits(Sample &s, std::vector<int> workload_map,
          std::vector<int> storage[4], int rounds);

void test_reduction_communication_computation(Sample &s,std::vector<int> workload_map,
          std::vector<int> storage[4], std::vector<int> storage_map[4], int rounds );



          void aggregate(vector<int> &out, vector<int> &in,
                  vector<long> &indptr, vector<long> &indices);
                  
          void shuffle(vector<long>& from_ids,  vector<int> &to,
                   vector<int>& from,  int start, int end);
          void pull_own_node(BiPartite *bp,
                vector<int> &out, vector<int> &in);
