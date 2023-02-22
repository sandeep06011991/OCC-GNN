 //  Cost of Traversal Cost of Computation,
 #include "../graph/dataset.cuh"
 #include "device_vector.h"
 #include <memory>
template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void naive_traversal(long * indptr, long * indices,\
          long num_nodes, \
   long num_edges, long* edge_color){
    int start = threadIdx.x + (blockIdx.x * TILE_SIZE);
    int end = min(static_cast<int64_t>(threadIdx.x + (blockIdx.x + 1) * TILE_SIZE), num_nodes);
    while(start < end){
      long num_nbs = indptr[start+1] - indptr[start];
      for(long i = 0; i <num_nbs; i++){
        long edge_id = indptr[start] + i;
        edge_color[edge_id] = start + indices[edge_id];
      }
      start +=  BLOCK_SIZE;
    }
}

#define WARP_SIZE (32)
// Applying GNN Advisor optimization in real time.
template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void edge_traversal(long * indptr, long * indices, long num_nodes, long num_edges, long* edge_color){
    int start = (blockIdx.x * TILE_SIZE);
    int end = (blockIdx.x + 1) * TILE_SIZE;

    __shared__ long indptrSh[BLOCK_SIZE + 1];
    while(start < end){
      __syncthreads();
      indptrSh[threadIdx.x] = indptr[start + threadIdx.x];

      // Edge cases ignored for now.
      if(threadIdx.x == 0){
        indptrSh[BLOCK_SIZE] = indptr[start + BLOCK_SIZE];
      }
      __syncthreads();
      // long start_edge = threadIdx.x;
      // long total_edge;
      // long start_offset;
      // if(threadIdx.x % WARP_SIZE == 0){
      //   long total_edges = indptrSh[BLOCK_SIZE] - indptrSh[0];
      //   start_offset = indptrSh[0];
      // }
      // total_edge = __shfl_sync(0xffffffff, total_edge, 0);
      // Edge parallel computation

      long nodeId = 0;
      long block_edge_start = indptrSh[0] + threadIdx.x;
      long block_edge_end = indptrSh[BLOCK_SIZE];
      while(block_edge_start < block_edge_end){
      //   long edge_offset = start_offset + start_edge;
      //   // But shared memory conflicts Different problem
      //   // But rarely if lots of no nodes occur together.
      //   // Divergance but colasced global memroy access
        while(indptrSh[nodeId  + 1] <=     block_edge_start){
          // Corectness before performance.
          nodeId +=1 ;

        }

        edge_color[block_edge_start] = indices[block_edge_start] + start + nodeId;
        block_edge_start +=  BLOCK_SIZE;
      }

      start += BLOCK_SIZE;;
    }
}

template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void  baseline(long * edges, long num_edges){
  int start = threadIdx.x + (blockIdx.x * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (blockIdx.x + 1) * TILE_SIZE), num_edges);
  while(start < end){
      edges[start]  = 1;
      start = start + TILE_SIZE;
  }
}

template<int BLOCK_SIZE, int TILE_SIZE>
__global__
void correct(long * color1, long * color2, long num_edges){
  int start = threadIdx.x + (blockIdx.x * TILE_SIZE);
  int end = min(static_cast<int64_t>(threadIdx.x + (blockIdx.x + 1) * TILE_SIZE), num_edges);
  while(start < end){
      if(color1[start] != color2[start]){
        printf("%ld %ld\n", color1[start], color2[start]);

      }
      assert(color1[start] == color2[start]);
      start = start + BLOCK_SIZE;
  }
}

void traversal(std::shared_ptr<Dataset> dataset){
  cuslicer::device_vector<long> color1;
  color1.resize(dataset->num_edges);
  cuslicer::device_vector<long> color2;
  color2.resize(dataset->num_edges);
  cudaEvent_t event1;
  cudaEvent_t event2;
  cudaEvent_t event3;
  cudaEvent_t event4;
  gpuErrchk(cudaEventCreate(&event1));
  gpuErrchk(cudaEventCreate(&event2));
  gpuErrchk(cudaEventCreate(&event3));
  gpuErrchk(cudaEventCreate(&event4));
  cudaEventRecord(event1);
  dataset->num_nodes = BLOCK_SIZE * 100;
  baseline<BLOCK_SIZE, TILE_SIZE><<<10000, BLOCK_SIZE>>>(color1.ptr(), dataset->num_edges);
  cudaEventRecord(event2);
  // gpuErrchk(cudaDeviceSynchronize());
  naive_traversal<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(dataset->num_nodes)-1 ,BLOCK_SIZE>>>\
      (dataset->indptr_d.ptr(), dataset->indices_d.ptr(),\
        dataset->num_nodes, dataset->num_edges, color1.ptr());
  cudaEventRecord(event3);
  // gpuErrchk(cudaDeviceSynchronize());
  edge_traversal<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(dataset->num_nodes) - 1,BLOCK_SIZE>>>\
      (dataset->indptr_d.ptr(), dataset->indices_d.ptr(),\
        dataset->num_nodes, dataset->num_edges, color2.ptr());
  cudaEventRecord(event4);
  gpuErrchk(cudaEventSynchronize(event4));
  float t1,t2;
  cudaEventElapsedTime ( &t1,  event2, event3 );
  cudaEventElapsedTime ( &t2,  event3, event4 );
  std::cout << "Naive :" << t1 <<"\n";
  std::cout << "Edgebased :" << t2 <<"\n";
  correct<BLOCK_SIZE, TILE_SIZE><<<GRID_SIZE(dataset->num_nodes)-1, BLOCK_SIZE>>>(color2.ptr(), \
        color1.ptr(), dataset->num_edges);
  gpuErrchk(cudaDeviceSynchronize());

}
