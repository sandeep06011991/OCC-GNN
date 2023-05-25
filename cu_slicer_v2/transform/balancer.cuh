#pragma once
#include "../util/device_vector.h"
#include "../util/cuda_utils.h"
#include "../util/types.h"
#include "../util/array_utils.h"
#include <curand.h>
#include <curand_kernel.h>

namespace cuslicer{

    __inline__  __device__ 
    PARTITIONIDX find_partition(float * prob_matrix, int num_gpus, float rand){
        float s = 0;
        for(PARTITIONIDX i = 0; i <  num_gpus; i++){
            s += prob_matrix[i];
            if(rand <= s) return i;
        }
        printf("prob found %f %f\n", rand, s);
        assert(false);
    }

    template<int BLOCKSIZE, int TILESIZE>
    __global__
    void rebalance_partitions(PARTITIONIDX *original_map, \
            size_t num_nodes, \
            PARTITIONIDX * new_map, float * prob_matrix,\
            curandState *random_states,
                 size_t num_random_states, int num_gpus){

    __shared__ unsigned char shRand[BLOCK_SIZE * sizeof(curandState)];


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
                if(prob_matrix[num_gpus * gpu + gpu] == 1.0){
                    new_map[tid] = gpu;
                }else{
                    float f = curand_uniform(curandSrcPtr ) ;
                    
                    new_map[tid] = find_partition\
                        (&prob_matrix[num_gpus * gpu], 
                            num_gpus, f);
                }
                start += BLOCK_SIZE;
            }
            tileId += gridDim.x;
        }
    }

    // Usually one partition is heavily loaded. 
    class LoadBalancer{

        float probability_matrix[MAX_DEVICES * MAX_DEVICES];
        // probabiliy after resampling marked from x to y 
        float * probability_matrix_d;
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
            cudaMalloc(&probability_matrix_d, MAX_DEVICES * MAX_DEVICES * sizeof(float));
        }

        void balance(device_vector<PARTITIONIDX> &workload_map, 
            device_vector<NDTYPE> &sample_in, 
                device_vector<PARTITIONIDX> &sample_workload_map, size_t last_layer_dst_nodes){

                    std::vector<NDTYPE> load; 
                    NDTYPE avg = 0;
                    std::cout << "attempt count \n";
                    for(int gpu = 0;gpu < this->num_gpus; gpu ++){
                        auto v =  count_if(sample_workload_map,  this->temp, gpu, last_layer_dst_nodes);
                        avg += v;
                        std::cout << "gpu " << gpu << ":" << v <<"\n";
                        load.push_back(v);
                    }
                    avg = avg/load.size();
                    memset(probability_matrix, 0, sizeof(float) * this->num_gpus * this->num_gpus);
                    for(int gpu = 0; gpu < num_gpus; gpu ++ ){
                        if(load[gpu] < avg){
                            probability_matrix[gpu * num_gpus + gpu] = 1;
                        }else{
                            auto excess_nodes = load[gpu] - avg;
                            auto total_moved = 0;
                            for(int move_to_gpu = 0; move_to_gpu < num_gpus; move_to_gpu ++ ){
                                if(excess_nodes <= 0)break;
                                if((move_to_gpu == gpu)  || (load[move_to_gpu] >= avg)) continue;
                                auto space = avg - load[move_to_gpu];
                                auto moved = min(space, excess_nodes);
                                probability_matrix[gpu * num_gpus  + move_to_gpu] = ((float) moved)/load[gpu];
                                excess_nodes -= moved;
                                total_moved += moved;
                                load[move_to_gpu] += moved;
                            }
                            probability_matrix[gpu * num_gpus + gpu] = ((float)(load[gpu] - total_moved))/load[gpu];
                            // load[gpu] -= total_moved
                            // Not required as we dont want to move anything to these partitions.

                        }
                        
                    }
                    std::cout << "probabiliy matrix\n";
                    for(int gpu = 0; gpu < num_gpus ; gpu ++ ){

                        for(int gpu1 = 0; gpu1 < num_gpus ; gpu1 ++ ){
                           std::cout <<  probability_matrix[gpu * num_gpus + gpu1] <<" "; 
                        }
                        std::cout << "\n";
                    }    
                    cudaMemcpy(probability_matrix_d, probability_matrix, sizeof(float) * MAX_DEVICES * MAX_DEVICES, cudaMemcpyHostToDevice);
                    rebalance_partitions<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(last_layer_dst_nodes),BLOCK_SIZE>>>\
                            (sample_workload_map.ptr(), last_layer_dst_nodes, \
                                sample_workload_map.ptr(), probability_matrix_d, \
                                 random_states, num_random_states,  num_gpus);
                    cudaDeviceSynchronize();
                    for(int gpu = 0;gpu < this->num_gpus; gpu ++){
                        auto v =  count_if(sample_workload_map,  this->temp, gpu, last_layer_dst_nodes);
                        avg += v;
                        std::cout << "gpu " << gpu << ":" << v <<"\n";
                        load.push_back(v);
                    }
                    // assert(false);             
                    std::cout << "Rebalancing done\n";
             }

    };


};