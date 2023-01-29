#pragma once
#include<vector>
#include "graph/sample.h"
#include "graph/sliced_sample.h"
#include <thrust/device_vector.h>

void test_sample_partition_consistency(Sample &s,
        PartitionedSample &p,
      thrust::device_vector<int> storage_map[8], int gpu_capacity[8], int num_nodes, int num_gpus);

      void aggregate(thrust::device_vector<int> &out, thrust::device_vector<int> &in,
              thrust::device_vector<long> &indptr, thrust::device_vector<long> &indices);

      void shuffle(thrust::device_vector<long>& from_ids,  thrust::device_vector<long> &to,
               thrust::device_vector<long>& from,  int start, int end);
      void pull_own_node(BiPartite *bp,
            thrust::device_vector<int> &out, thrust::device_vector<int> &in);

/*
void test_sample_partition_consistency_gat(Sample &s,
              PartitionedSample &p,
            std::vector<int> storage_map[8], int gpu_capacity[8], int num_nodes, int num_gpus);

void  test_pull_benefits(Sample &s, std::vector<int> workload_map,
          std::vector<int> storage[4], int rounds);

void test_reduction_communication_computation(Sample &s,std::vector<int> workload_map,
          std::vector<int> storage[4], std::vector<int> storage_map[4], int rounds );


      */
