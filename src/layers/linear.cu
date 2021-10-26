#include "tensor.hh"
#include "linear.hh"
#include "nn_exception.hh"
#include <assert.h>
#include <iostream>

LinearLayer::LinearLayer(int dim1, int dim2){
  this->dim1 = dim1;
  this->dim2 = dim2;
  this->W = new Tensor<float>(allocate_random(dim1*dim2,dim2),dim1,dim2);
  this->b = new Tensor<float>(allocate_random(dim2,dim2),dim2,1);
  this->dW = new Tensor<float>(dim1,dim2);
  this->db = new Tensor<float>(dim2,1);

  this->in = *(new Tensor<float>(1,1));
}

LinearLayer::LinearLayer(float *W, float *B, int dim1, int dim2){
  this->dim1 = dim1;
  this->dim2 = dim2;
  this->W = new Tensor<float>(W,dim1,dim2);
  this->b = new Tensor<float>(B,dim2,1);
  this->dW = new Tensor<float>(dim1,dim2);
  this->db = new Tensor<float>(dim2,1);
  this->in = *(new Tensor<float>(1,1));
}

LinearLayer::LinearLayer(int dim1, int dim2, int in_dim){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->in_dim = in_dim;
    this->out = new Tensor<float>(in_dim,dim2);
    this->W = new Tensor<float>(allocate_random(dim1*dim2,dim2),dim1,dim2);
    this->b = new Tensor<float>(allocate_random(dim2,dim2),dim2,1);
    this->out_grad = new Tensor<float>(in_dim,dim1);
    this->dW = new Tensor<float>(allocate_random(dim1*dim2,dim2),dim1,dim2);
    this->db = new Tensor<float>(allocate_random(dim2,dim2),dim2,1);
    this->_btemp =  allocate_ones(in_dim,1);
    this->in = *(new Tensor<float>(1,1));

}


  void LinearLayer::update(float learning_rate){
    // std::cout << "Checking gradients \n";
    // this->dW->debugTensor();
    // this->dW->viewTensor();
    this->W->update(learning_rate, this->dW);
    this->b->update(learning_rate, this->db);
  }

LinearLayer::LinearLayer(float *W, float *B, int dim1, int dim2, int in_dim){
    this->dim1 = dim1;
    this->dim2 = dim2;
    this->in_dim = in_dim;
    this->out = new Tensor<float>(in_dim,dim2);
    this->W = new Tensor<float>(W,dim1,dim2);
    this->b = new Tensor<float>(B,dim2,1);
    this->out_grad = new Tensor<float>(in_dim,dim1);
    this->dW = new Tensor<float>(allocate_random(dim1*dim2,dim2),dim1,dim2);
    this->db = new Tensor<float>(allocate_random(dim2,dim2),dim2,1);
    this->_btemp =  allocate_ones(in_dim,1);
    this->in = *(new Tensor<float>(1,1));
}



// Copied code modifu this later to get grads as well
__global__ void cu_multiply(float* A, float* B, float * C,
                                    int rowsa, int colsa,
                                    int rowsb, int colsb,
                                    int rowsc, int colsc){
// Completely wrong. Redo this.
    __shared__ float sA[32][32];   // Tile size of 32x32
    __shared__ float sB[32][32];
    int Row = blockDim.x*blockIdx.x + threadIdx.x;
    int Col = blockDim.y*blockIdx.y + threadIdx.y;
    float Cvalue = 0.0;
    sA[threadIdx.x][threadIdx.y] = 0.0;
    sB[threadIdx.x][threadIdx.y] = 0.0;

    for (int k = 0; k < (((colsa - 1)/ 32) + 1); k++){
        if ( (Row < rowsa) && (threadIdx.y + (k*32)) < colsa){
            sA[threadIdx.x][threadIdx.y] = A[(Row*colsa) + threadIdx.y + (k*32)];
        }
        else{
            sA[threadIdx.x][threadIdx.y] = 0.0;
        }
        __syncthreads();
        if ( Col < colsb && (threadIdx.x + k*32) < rowsb){
          // Jumping . move access
            sB[threadIdx.x][threadIdx.y] = B[(threadIdx.x + k*32)*colsb + Col];
        }
        else{
            sB[threadIdx.x][threadIdx.y] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < 32; ++j){
            Cvalue += sA[threadIdx.x][j] * sB[j][threadIdx.y];
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

__global__ void cu_add_bias(float *out, float *bias){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
	out[tid] = out[tid] + bias[threadIdx.x];
}

Tensor<float>& LinearLayer::forward(Tensor<float>& in_p){

  this->in_dim = in_p.dim1;
  this->in = in_p;
  Tensor<float>& in = in_p;
  int TILE_WIDTH = 32;

  // in.debugTensor();
  // W->debugTensor();
  if(this->out !=nullptr){
    delete this->out;
    delete this->out_grad;
    delete this->_btemp;
  }
  this->out = new Tensor<float>(in_dim,dim2);
  this->out_grad = new Tensor<float>(in_dim,dim1);
  this->_btemp =  allocate_ones(in_dim,1);

  mat_mul_a_b(in, true, *this->W, true , *this->out);
  NNException::throwIfDeviceErrorsOccurred("mat mul   failed");

  cu_add_bias<<<out->dim1, out->dim2>>>(out->data_device, b->data_device);
  NNException::throwIfDeviceErrorsOccurred("mat mul linear failed");

  cudaDeviceSynchronize();
  return *out;
}

// void compute_w_grad(float *in, in_dim1, in_dim2, float *wx)

Tensor<float>& LinearLayer::backward(Tensor<float>& in_grad){
  assert(in_grad.dim1 = this->in_dim );
  assert(in_grad.dim2 = this->dim2);
  mat_mul_a_t_b(this->in,true, in_grad, true, *dW);
  // Tensor<float> * out_grad;
  mat_mul_a_b_t(in_grad,true,*W,true,*out_grad);
  mat_mul_a_t_b(in_grad,true,*this->_btemp,true,*db);
  cudaDeviceSynchronize();
    // compute_w_grad(in_grad);
  return *out_grad;
}
