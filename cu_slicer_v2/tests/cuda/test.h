#pragma once
#include<vector>
#include "../../graph/sample.h"
#include "../../graph/sliced_sample.h"
#include "../../util/device_vector.h"
#include "../../util/types.h"

using namespace cuslicer;

void test_sample_partition_consistency(Sample &s,
    PartitionedSample &p,
   std::vector<NDTYPE> storage_map[8], int gpu_capacity[8], int num_nodes, int num_gpus);

void aggregate(device_vector<int> &out, device_vector<int> &in,
        device_vector<long> &indptr, device_vector<long> &indices);

void shuffle(device_vector<long>& from_ids, device_vector<long> &to,
           device_vector<long>& from,  int start, int end);

void pull_own_node(BiPartite *bp,
      device_vector<int> &out, device_vector<int> &in);

/*
void test_sample_partition_consistency_gat(Sample &s,
              PartitionedSample &p,
            std::vector<int> storage_map[8], int gpu_capacity[8], int num_nodes, int num_gpus);

void  test_pull_benefits(Sample &s, std::vector<int> workload_map,
          std::vector<int> storage[4], int rounds);

void test_reduction_communication_computation(Sample &s,std::vector<int> workload_map,
          std::vector<int> storage[4], std::vector<int> storage_map[4], int rounds );


      */
