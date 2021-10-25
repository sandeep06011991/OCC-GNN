// Naive global aggregation
#include "tensor.hh"
#include "gnn/sage.hh"
#include "nn_exception.hh"
#include <iostream>

__global__ void aggregate_nodeWise(float *inFeat, float *outFeat, int *offsets,
    int *indices, int fsize){
      int outNodeID = blockIdx.x;
      float t = 0;
      for(int edge_off = offsets[outNodeID]; edge_off < offsets[outNodeID + 1]; edge_off ++ ){
        int edge_id = indices[edge_off];
        t += inFeat[edge_id * fsize + threadIdx.x];
      }
      outFeat[outNodeID * fsize + threadIdx.x] = t;

}

__global__ void  aggregate_edgeWise(float *ingrad, float *outgrad, int *offsets,
    int *indices, int fsize){
      int outNodeID = blockIdx.x;
      float t = ingrad[outNodeID* fsize + threadIdx.x];;
      for(int edge_off = offsets[outNodeID]; edge_off < offsets[outNodeID + 1]; edge_off ++ ){
        int edge_id = indices[edge_off];
        atomicAdd(&outgrad[edge_id * fsize + threadIdx.x],t);
      }
}


Tensor<float>& SageAggr::forward(Tensor<int>& offsets , Tensor<int>& indices,
        Tensor<float>& in, int num_nodes_out, int num_nodes_in){
    if(this->out_feat != nullptr){
      delete this->out_feat;
    }
    if(this->out_grad != nullptr){
      delete this->out_grad;
    }
    this->out_feat = new Tensor<float>(num_nodes_out,this->fsize);
    this->out_grad = new Tensor<float>(num_nodes_in,this->fsize);
    int blocks = num_nodes_out;
    int threads = fsize;
    this->num_nodes_out = num_nodes_out;
    this->num_nodes_in = num_nodes_in;
    this->offsets = offsets ;
    this->indices = indices;

    aggregate_nodeWise<<<blocks, threads>>>(in.data_device, out_feat->data_device, \
      offsets.data_device, indices.data_device, this->fsize);
    std::cout << "Forward pass kernels launched \n";
    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred("forward Pass failed");
    return *out_feat;
}

Tensor<float>& SageAggr::backward(Tensor<float>& in_grad){
  int blocks = this->num_nodes_out;
  int threads = this->fsize;
  aggregate_edgeWise<<<blocks, threads>>>(in_grad.data_device, this->out_grad->data_device, \
    this->offsets.data_device, this->indices.data_device, this->fsize);
  cudaDeviceSynchronize();
  NNException::throwIfDeviceErrorsOccurred("backward pass failed\n");
  return *out_grad;
}
