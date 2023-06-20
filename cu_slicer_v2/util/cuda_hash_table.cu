#include "device_vector.h"
#include "cuda_hash_table.cuh"
#include <cub/cub.cuh>

using namespace cuslicer;

template <typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(const IdType running_total)
      : running_total_(running_total) {}

  __device__ IdType operator()(const IdType block_aggregate) {
    const IdType old_prefix = running_total_;
    running_total_ += block_aggregate;
    return old_prefix;
  }
};

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_duplicates(
    const IdType* const items, const int64_t num_items,
    DeviceOrderedHashTable table) {
        assert(BLOCK_SIZE == blockDim.x);
        const size_t block_start = TILE_SIZE * blockIdx.x;
        const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
        if (index < num_items) {
            table.Insert(items[index], index);
        }
    }
}


template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void replace( IdType * items, const int64_t num_items,
    DeviceOrderedHashTable table) {
        assert(BLOCK_SIZE == blockDim.x);
        const size_t block_start = TILE_SIZE * blockIdx.x;
        const size_t block_end = TILE_SIZE * (blockIdx.x + 1);
        using Mapping = typename DeviceOrderedHashTable::Mapping;

    #pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end;
        index += BLOCK_SIZE) {
        if (index < num_items) {
            const Mapping * mapping = table.Search(items[index]);
            items[index] = mapping->local;
        }
    }
}

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(
    const IdType* items, const size_t num_items,
    DeviceOrderedHashTable table, IdType* const num_unique) {
        assert(BLOCK_SIZE == blockDim.x);

        using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
        using Mapping = typename DeviceOrderedHashTable::Mapping;

        const size_t block_start = TILE_SIZE * blockIdx.x;
        const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

        IdType count = 0;

        #pragma unroll
        for (size_t index = threadIdx.x + block_start; index < block_end;
            index += BLOCK_SIZE) {
            if (index < num_items) {
            const Mapping& mapping = *table.Search(items[index]);
            if (mapping.index == index) {
                ++count;
            }
            }
        }

        __shared__ typename BlockReduce::TempStorage temp_space;

        count = BlockReduce(temp_space).Sum(count);

        if (threadIdx.x == 0) {
            num_unique[blockIdx.x] = count;
            if (blockIdx.x == 0) {
                num_unique[gridDim.x] = 0;
            }
        }
    }


template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(
    const IdType* const items, const size_t num_items,
    DeviceOrderedHashTable table,
    const IdType* const num_items_prefix, IdType* const unique_items,
    int64_t* const num_unique_items) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable::Mapping;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    Mapping* kv;
    if (index < num_items) {
      kv = (Mapping *)table.Search(items[index]);
      flag = kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = offset + flag;
      kv->local = pos;
      unique_items[pos] = items[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = num_items_prefix[gridDim.x];
  }
}
  
 void OrderedHashTable::FillWithDuplicates(\
      const NDTYPE* input, const size_t num_input, NDTYPE* const unique,\
         int64_t* const num_unique){
  
    const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);

    (generate_hashmap_duplicates<NDTYPE, BLOCK_SIZE, TILE_SIZE><<<grid, block>>>\
        (input, num_input, this->_table));
    gpuErrchk(cudaDeviceSynchronize());
    
    device_vector<NDTYPE> item_prefix;
    item_prefix.resize(num_input + 1);
    
    (count_hashmap<NDTYPE, BLOCK_SIZE, TILE_SIZE><<<grid, block>>>\
        (input, num_input, this->_table, item_prefix.ptr()));
    
    gpuErrchk(cudaDeviceSynchronize());
    size_t workspace_bytes;
    
    gpuErrchk(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<NDTYPE *>(nullptr),
      static_cast<NDTYPE *>(nullptr), grid.x + 1));
    
    device_vector<NDTYPE> d_temp_storage;
    d_temp_storage.resize(workspace_bytes/(sizeof(NDTYPE)) + 1);
    gpuErrchk(cudaDeviceSynchronize());  
    
    gpuErrchk(cub::DeviceScan::ExclusiveSum(
            d_temp_storage.ptr(), workspace_bytes, item_prefix.ptr(), item_prefix.ptr(), grid.x + 1));
    
    gpuErrchk(cudaDeviceSynchronize());
    compact_hashmap<NDTYPE, BLOCK_SIZE, TILE_SIZE><<<grid, block>>>(\
            input, num_input, this->_table, item_prefix.ptr(), unique, num_unique);
    gpuErrchk(cudaDeviceSynchronize());
}


  void OrderedHashTable::Replace(NDTYPE * input, const size_t num_input){
   const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);
    (replace<NDTYPE, BLOCK_SIZE, TILE_SIZE><<<grid, block>>>\
        (input, num_input, this->_table));
    gpuErrchk(cudaDeviceSynchronize());
  }