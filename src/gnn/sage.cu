// Naive global aggregation
#include "util/tensor.hh"
#include "gnn/sage.hh"
#include "nn_exception.hh"
#include <iostream>
#include "util/timer.h"
__global__ void aggregate_nodeWise(float *inFeat, float *outFeat, int *offsets,
    int *indices, int fsize){
      int outNodeID = blockIdx.x;
      float t = 0;
      for(int edge_off = offsets[outNodeID]; edge_off < offsets[outNodeID + 1];
        edge_off ++ ){
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


Tensor<float> * SageAggr::forward(Tensor<int>& offsets , Tensor<int>& indices,
        Tensor<float>& in, int num_nodes_out, int num_nodes_in){

    assert(in.s.dim1 == num_nodes_in);
    assert(offsets.s.dim1 == num_nodes_out + 1);
    if(this->out_feat != nullptr){
      this->out_feat->clearTensor();
      delete this->out_feat;
    }
    if(this->out_grad != nullptr){
      this->out_grad->clearTensor();
      delete this->out_grad;
    }
    // start_timer(MOVEMENT_COMPUTE1);
    this->out_feat = new Tensor<float>(Shape(num_nodes_out,this->fsize),this->device_id);
    this->out_grad = new Tensor<float>(Shape(num_nodes_in,this->fsize),this->device_id);
    int blocks = num_nodes_out;
    int threads = this->fsize;
    this->num_nodes_out = num_nodes_out;
    this->num_nodes_in = num_nodes_in;
    this->offsets = offsets ;
    this->indices = indices;
    cudaSetDevice(this->device_id);
    NNException::throwIfDeviceErrorsOccurred("forward Pass pre failed");
   auto  error = cudaEventRecord(start);
    aggregate_nodeWise<<<blocks, threads>>>(in.data_device, out_feat->data_device, \
      offsets.data_device, indices.data_device, this->fsize);
    auto error1 = cudaEventRecord(stop);
    auto error2 = cudaEventSynchronize(stop);
    float msec = 0.0f;
    auto error3 = cudaEventElapsedTime(&msec, this->start, this->stop);
    add_timer_ms(MOVEMENT_COMPUTE1,msec);
    // stop_timer(MOVEMENT_COMPUTE1);
    // cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred("forward Pass failed");
    return out_feat;
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
