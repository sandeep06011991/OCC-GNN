#pragma once

#include "tensor.hh"
#include "linear.hh"
#include "nn_exception.hh"


LinearLayer::LinearLayer(int dim1, int dim2, int in_dim){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->in_dim = in_dim;
    this->out = new Tensor<float>(in_dim,dim2);
    this->W = new Tensor<float>(allocate_random(dim1*dim2),dim1,dim2);
    this->b = new Tensor<float>(allocate_random(dim2),dim2,1);
    this->in_grad = new Tensor<float>(in_dim,dim1);
}

// Copied code modifu this later to get grads as well
__global__ void cu_multiply(float* A, float* B, float * C,
                                    int rowsa, int colsa,
                                    int rowsb, int colsb,
                                    int rowsc, int colsc){
    __shared__ float sA[32][32];   // Tile size of 32x32
    __shared__ float sB[32][32];
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;
    for (int k = 0; k < (((colsa - 1)/ 32) + 1); k++){
        if ( (Row < rowsa) && (threadIdx.x + (k*32)) < colsa){
            sA[threadIdx.y][threadIdx.x] = A[(Row*colsa) + threadIdx.x + (k*32)];
        }
        else{
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
        if ( Col < colsb && (threadIdx.y + k*32) < rowsb){
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*32)*colsb + Col];
        }
        else{
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < 32; ++j){
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (Row < rowsc && Col < colsc){
        C[Row*colsc + Col] = Cvalue;
    }
}

// CUDA PLAS
// a += b
// n is size of a
__global__ void cu_plus(float *A, const float *B, const int n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while(tid < n){
		A[tid] = __fadd_rd(A[tid], B[tid]);
		tid += stride;
	}
}

Tensor<float>& LinearLayer::computeForwardPass(Tensor<float>& in){
  int TILE_WIDTH = 32;
  dim3 dimGrid((out->dim2 - 1) / TILE_WIDTH + 1, (out->dim1 - 1) / TILE_WIDTH + 1, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  cu_multiply<<<dimGrid, dimBlock>>>((float*)in.data_device ,
                (float *) W->data_device,
                (float *) out->data_device,
								in.dim1, in.dim2,
                W->dim1, W->dim2,
                out->dim1, out->dim2 );
  NNException::throwIfDeviceErrorsOccurred("mat mul linear failed");
  cudaDeviceSynchronize();
  return *out;
}


Tensor<float>& LinearLayer::computeBackwardPass(Tensor<float>& in_grad){
  return *out_grad;
}
