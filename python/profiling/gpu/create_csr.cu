
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <cstdlib>
#include <thrust/scan.h>
#include <iostream>

struct edge{
  int a;
  int b;
};

__host__ __device__ bool operator<(const edge &lhs, const edge &rhs)
{
 if(lhs.a == rhs.a) return lhs.b < rhs.b;
 return lhs.a < rhs.b;
 // return lhs.distance < rhs.distance;
}

__global__ void diff(struct edge * start,int * t1){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i==0){
    t1[i] = 1;
    return;
  }
  if(start[i].a == start[i-1].a){
    t1[i] = 0;
  } else{
    t1[i] = 1;
  }
}


__global__ void populate_offset(struct edge * edges,
       int *unq, int *unq_pos,
        int * nodes, int *offsets){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(unq[i]==1){
    nodes[unq_pos[i]] = edges[i].a;
    offsets[unq_pos[i]] = i;
  }
}


__global__ void diff_offsets(int *offsets){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    offsets[i] = offsets[i+1]-offsets[i];
  }

__global__ void populate_node_degree(int *nodes, int *offsets, int *nd_degree){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  nd_degree[nodes[i]] = offsets[i];
}

int main(void)
{
  int n = 6;
  int e = 6;
  int e1[] = {1,2,3,5,5,3};
  int e2[] = {2,3,4,5,8,0};
  thrust::host_vector<edge> h_vec(6);
  // std::vector<edge> v;
  for(int i=0;i<6;i++){
    // struct edge e;
    h_vec[i].a = e1[i];
    h_vec[i].b = e2[i];
  }
  thrust::device_vector<edge> d_vec = h_vec;
  thrust::device_vector<int> unq(e);
  thrust::device_vector<int> unq_pos(e);

  thrust::sort(d_vec.begin(), d_vec.end());

  struct edge * ev = thrust::raw_pointer_cast(&d_vec[0]);
  diff<<<1,6>>>(ev,thrust::raw_pointer_cast(&unq[0]));
  cudaDeviceSynchronize();

  int unq_nodes = thrust::reduce(unq.begin(),unq.end());
  thrust::exclusive_scan(unq.begin(), unq.end(),unq_pos.begin());

  thrust::device_vector<int> nodes(unq_nodes);
  thrust::device_vector<int> offsets(unq_nodes+1);

  thrust::device_vector<int> nd_degree(n+1);

  populate_offset<<<1,e>>>(ev, thrust::raw_pointer_cast(&unq[0]),
            thrust::raw_pointer_cast(&unq_pos[0]), thrust::raw_pointer_cast(&nodes[0]),
            thrust::raw_pointer_cast(&offsets[0]));
  cudaDeviceSynchronize();
  offsets[unq_nodes] = e;

  thrust::fill(nd_degree.begin(),nd_degree.end(),0);
  diff_offsets<<<1,unq_nodes>>>(thrust::raw_pointer_cast(&offsets[0]));
  populate_node_degree<<<1,unq_nodes>>>(thrust::raw_pointer_cast(&nodes[0]),
              thrust::raw_pointer_cast(&offsets[0]),
                thrust::raw_pointer_cast(&nd_degree[0]));
  thrust::exclusive_scan(nd_degree.begin(),nd_degree.end(), nd_degree.begin());
  for(int i=0;i<nd_degree.size();i++){
    std::cout << i << " " << nd_degree[i] <<"\n";
  }
}
