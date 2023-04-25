#pragma once
#include "../util/device_vector.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"
#include "../util/array_utils.h"
#include <curand.h>
#include <curand_kernel.h>

namespace cuslicer{


    PARTITIONIDX find_partition(float * prob_matrix, int num_gpus, float rand){
        auto s = 0;
        for(int i = 0; i <  num_gpus; i++){
            s += prob_matrix[i];
            if(rand < s) return i;
        }
    }

    template<int BLOCKSIZE, int TILESIZE>
    __global__
    void rebalance_partitions(NDTYPE *original_map, \
            size_t num_nodes, \
            NDTYPE * new_map, float ** prob_matrix,\
            curandState *random_states,
                 size_t num_random_states){
    const int intsInRandState = sizeof(curandState)/sizeof(int);
    int* shStateBuff = (int*)&shRand[0];

    int* randStatesAsInts = (int*)random_states;

    for (int i = threadIdx.x; i < intsInRandState*blockDim.x; i += blockDim.x) {
      shStateBuff[i] = randStatesAsInts[i + blockDim.x*blockIdx.x];
    }

    __syncthreads();
    auto curandSrcPtr = (curandState*)(&shStateBuff[threadIdx.x*intsInRandState]);
        int tileId = blockIdx.x;
        int last_tile = (( num_nodes - 1) / TILE_SIZE + 1);
        while(tileId < last_tile){
            int start = threadIdx.x + (tileId * TILE_SIZE);
            int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), num_nodes);
            while(start < end){
                auto tid = start;
                auto gpu = original_map[tid];
                if(prob_matrix[gpu][gpu] == 1.0){
                    new_map[tid] = gpu;
                }else{
                    float f = curand_uniform(curandSrcPtr ) ;
                    new_map[tid] = find_partition(prob_matrix, num_gpus, rand);
                }
                start += BLOCK_SIZE;
            }
            tileId += gridDim.x;
        }
    }

    // Usually one partition is heavily loaded. 
    class LoadBalancer{

        float probability_matrix[MAX_DEVICES][MAX_DEVICES];
        // probabiliy after resampling marked from x to y 

        curandState * random_states;
        size_t num_random_states;

        device_vector<PARTITIONIDX> partition_index;
        device_vector<PARTITIONIDX> temp;

        // Random map 
        int num_gpus;
public:
        LoadBalancer(int num_devices,\
            curandState *random_states, size_t num_random_states){
            this->num_gpus = num_devices;
            this->num_random_states = num_random_states;
            this->random_states = random_states;
        }

        void balance(device_vector<int> &workload_map, 
            device_vector<int> &sample_in, 
                device_vector<int> &sample_workload_map){

                    
                    std::vector<NDTYPE> load; 
                    NDTYPE avg = 0;
                    for(int gpu = 0;gpu < this->num_gpus; gpu ++){
                        auto v =  count_if(workload_map,  this->temp, gpu);
                        avg += v;
                    }
                    avg = avg/load.size();
                    memset(probability_matrix, 0, sizeof(float) * this->num_gpus * this->num_gpus);
                    for(int gpu = 0; gpu < num_gpus; gpu ++ ){
                        if(load[gpu] < avg){
                            probability_matrix[gpu][gpu] = 1;
                        }else{
                            auto excess_nodes = load[gpu] - avg;
                            auto total_moved = 0;
                            for(int move_to_gpu = 0; move_to_gpu < num_gpus; move_to_gpu ++ ){
                                if(excess_nodes <= 0)break;
                                if((move_to_gpu == gpu)  || (load[move_to_gpu] >= avg)) continue;
                                auto space = avg - load[move_to_gpu];
                                auto moved = min(space, excess_nodes);
                                probability_matrix[gpu][move_to_gpu] = moved/load[gpu];
                                excess_nodes -= moved;
                                total_moved += moved;
                                load[move_to_gpu] += moved;
                            }
                            probability_matrix[gpu][gpu] = (load[gpu] - total_moved)/load[gpu];
                            // load[gpu] -= total_moved
                            // Not required as we dont want to move anything to these partitions.

                        }
                        
                    }
             }

    };


};