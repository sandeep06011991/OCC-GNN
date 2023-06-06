#pragma once
#include "device_vector.h"
#include "cuda_utils.h"

#include "device_vector.h"
#include "cub.h"
#include "../graph/order_book.h"
namespace cuslicer{


template<int BLOCK_SIZE, int TILE_SIZE, typename T1, typename T2>
__global__
void  index_in_kernel(T1 * in, size_t sz,
        OrderBook * orderbook, T2 * out){
        int tileId = blockIdx.x;
        int last_tile = ((sz - 1) / TILE_SIZE + 1);
        while(tileId < last_tile){
            int start = threadIdx.x + (tileId * TILE_SIZE);
            int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), sz);
            while(start < end){
                    out[start] = orderbook->findWorkloadPartition(in[start]);
                    start += BLOCK_SIZE;
                }
            tileId += gridDim.x;
        }
    }
// nodes, this->orderbook->getDevicePtr(), this->sample_workload_map
// out[tid] = index[in[tid]]
    template<typename T1, typename T2>
    void index_in(device_vector<T1>& input, OrderBook * orderbook, device_vector<T2>& out){
        index_in_kernel<BLOCK_SIZE, TILE_SIZE, T1, T2><<<GRID_SIZE(input.size()), BLOCK_SIZE>>>\
        (input.ptr(), input.size(), orderbook, out.ptr());
    }
    
template<int BLOCK_SIZE, int TILE_SIZE, typename T1>
__global__
void  mark_if(T1 * in, size_t sz,
        T1 * index, T1 val){
        int tileId = blockIdx.x;
        int last_tile = ((sz - 1) / TILE_SIZE + 1);
        while(tileId < last_tile){
            int start = threadIdx.x + (tileId * TILE_SIZE);
            int end = min(static_cast<int64_t>(threadIdx.x + (tileId + 1) * TILE_SIZE), sz);
            while(start < end){
                    if(in[start] == val){
                        index[start] = 1;
                    }else{
                        index[start] = 0;
                    }        
                    start += BLOCK_SIZE;
                }
            tileId += gridDim.x;
        }
    }
  

    template<typename T1>
    T1 count_if(device_vector<T1>& input,\
         device_vector<T1>& temp, T1 eq, size_t sz){
        // Set value if eq
        assert(input.size() >= sz);
        temp.resize(sz);

        mark_if<BLOCK_SIZE, TILE_SIZE, T1><<<GRID_SIZE(sz), BLOCK_SIZE>>>(
            input.ptr(), sz, temp.ptr(), eq
        );
        return transform<T1>::reduce(temp);
    }

};