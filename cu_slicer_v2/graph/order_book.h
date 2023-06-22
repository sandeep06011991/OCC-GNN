#pragma once 
#include "../util/types.h"
#include "../util/cuda_utils.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;


// Using constant Memory
class OrderBook {
    public:
    int partition_offsets[MAX_GPUS];
    int cached_offsets[MAX_GPUS][MAX_GPUS];
    int num_partitions;
    
    __inline__ __device__
    int findWorkloadPartition(NDTYPE nd){
        for(int i = 0; i < num_partitions; i++){
            if(nd < partition_offsets[i + 1]){
                return i;
            }
        }
        printf("%d %d\n", partition_offsets[4], nd);
        assert(false);
    }

    __inline__ __device__ 
    bool gpuContains(int partition,int nd2){
        for(int i = 0; i < num_partitions; i ++ ){
            if(partition_offsets[i + 1] > nd2){
                if(cached_offsets[partition][i]> nd2){
                    return true;
                }else{
                    return false;
                }
            }
        }
        // partition not found
        assert(false);
    }

    __inline__ __device__ 
    int getLocalId(int partition, int nd2){
        int offsets = 0;
        for(int i = 0; i < num_partitions; i++){
            if(partition_offsets[i + 1] > nd2){
                if(cached_offsets[partition][i]> nd2){
                    offsets += nd2 - partition_offsets[i];  
                    return offsets;
                }else{
                    assert(false);
                }
            }
            offsets += cached_offsets[partition][i] - partition_offsets[i];
        }
        assert(false);
        return -1;
    }

    OrderBook(std::string BIN_DIR, std::string graphname,\
                 std::string size, int partitions);


};