#pragma once
#include "../util/device_vector.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"
namespace cuslicer{

    __global__ 
    void flip(){

        if (partition is overloaded){
            // run random number 
            for(int i = 0 ; i < num_gpus; i++){
                if(// In interval flip);
            }
        }
    }
    // Usually one partition is heavily loaded. 
    class LoadBalancer{

        float probability_matrix[MAX_DEVICES][MAX_DEVICES];
        device_vector<NDTYPE> partition_index;
        // Random map 
    
public:
        LoadBalancer(int num_devices){

        }
        void balance(device_vector<int> &workload_map, 
            device_vector<int> &sample_in, 
                device_vector<int> &sample_workload_map){

                    // 1. Fill up partitiotn 

                    // 2. Get individual sums 

                    // 3. 
                    std::vector<NDTYPE> load; 
                    auto avg ;


                }

    };


};