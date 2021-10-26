#include "loss.hh"
#include "nn_exception.hh"
#include<assert.h>

// Copied code modifu this later to get grads as well
__global__ void cu_exponent(float* in, float* out, int size){
  int offset = blockDim.x*blockIdx.x + threadIdx.x;
  while(offset < size){
    out[offset] = expf(in[offset]);
    offset = offset + gridDim.x*blockDim.x;
  }
}

__global__ void cu_exponent_sum(float *in, float *out, int N, int noClasses){
  __shared__ float s[64];
  int n = blockIdx.x;
  assert(threadIdx.x < 64);
  if(threadIdx.x < noClasses){
    s[threadIdx.x] = in[n * noClasses + threadIdx.x];
    // s[threadIdx.x] = 200;
  }
  __syncthreads();
  // Bad Code here but dont worry now!
  if(threadIdx.x == 0){
    for(int i=1;i<noClasses;i++){
      s[0] = s[0] + s[i];
    }
    out[n] = s[0];
  }
  // int i = noClasses/2;
  // while(i!=0){
  //   if(threadIdx.x<i){
  //       s[threadIdx.x] = s[threadIdx.x + i] + s[threadIdx.x];
  //   }
  //   i = i/2;
  //   __syncthreads();
  // }
  // if(threadIdx.x == 0)
}

__global__ void cu_loss(float *x, int * y, float *exp_sum, float *loss, int N, int noClasses){
  int offset = blockDim.x*blockIdx.x + threadIdx.x;
  if(offset < N){
    loss[offset] = -x[offset * noClasses + y[offset]] + logf(exp_sum[offset]);
    //
  }
}

__global__ void cu_gradient(float *exp, float *exp_sum, int *y, float *out,  int N, int D){
  int n = blockIdx.x;
  int d = threadIdx.x;
  int global_id = n * blockDim.x + d;
  float t = exp[global_id]/exp_sum[n];
  if(threadIdx.x == y[n]){
    t = t - 1;
  }
  out[global_id] = t/N;
}

CrossEntropyLoss::CrossEntropyLoss(int n, int d){
  this->N = n;
  this->D = d;
  this->exp_x = new Tensor<float>(n,d);
  this->exp_sum = new Tensor<float>(n,1);
  this->loss = new Tensor<float>(n,1);
  this->dx = new Tensor<float>(n,d);
}

void CrossEntropyLoss::compute_exponent(Tensor<float> &in){
  int dim1 = in.dim1;
  int dim2 = in.dim2;
  int noBlocks = ((in.dim1 * in.dim2) + 256)/256;
  int noThreads = 256;
  cu_exponent<<<noBlocks,noThreads>>>((float *) in.data_device,
  (float *)  this->exp_x->data_device, dim1 * dim2);
  NNException::throwIfDeviceErrorsOccurred("compute exponent failed ");
}

void CrossEntropyLoss::compute_exponent_sum(){
  cu_exponent_sum<<<this->N,this->D>>>((float *) exp_x->data_device,
  (float *)  exp_sum->data_device, this->N, this-> D);
  NNException::throwIfDeviceErrorsOccurred("compute exponent sum failed ");
}

void CrossEntropyLoss::compute_loss(Tensor<float> &in,
                Tensor<int> &true_labels){
  int noBlocks = ((N + 256)/256);
  int noThreads = 256;
  cu_loss<<<noBlocks,noThreads>>>((float *)in.data_device, true_labels.data_device,
        (float *) exp_sum->data_device,  loss->data_device, N, D);
  NNException::throwIfDeviceErrorsOccurred("compute exponent loss failed ");
}

// For classification
// in Shape = N,C
// N num examples, C num examples.
// returns Tensor of shape N,1
Tensor<float>& CrossEntropyLoss::forward(Tensor<float> &in,Tensor<int> &true_labels){
    // in.debugTensor();
    this->N = in.dim1;
    this->D = in.dim2;
    assert(in.dim2 ==3);
    int dim1 = in.dim1;
    int dim2 = in.dim2;
    // in.debugTensor();
    // in.viewTensor();
    // true_labels.debugTensor();
    // true_labels.viewTensor();
    if(this->exp_x != nullptr){
      this->exp_x->cleanUpTensor();
      this->exp_sum->cleanUpTensor();
      this->loss->cleanUpTensor();
      this->dx->cleanUpTensor();
      delete this->exp_x;
      delete this->exp_sum;
      delete this->loss;
      delete this->dx;
      NNException::throwIfDeviceErrorsOccurred("compute deletion failed ");
    }
    this->exp_x = new Tensor<float>(dim1,dim2);
    this->exp_sum = new Tensor<float>(dim1,1);
    this->loss = new Tensor<float>(dim1,1);
    this->dx = new Tensor<float>(dim1,dim2);

    compute_exponent(in);
    NNException::throwIfDeviceErrorsOccurred("BCE Failed1 ");

    compute_exponent_sum();
    NNException::throwIfDeviceErrorsOccurred("BCE Failed2 ");

    compute_loss(in, true_labels);
    NNException::throwIfDeviceErrorsOccurred("BCE Failed3 ");

    cudaDeviceSynchronize();
    NNException::throwIfDeviceErrorsOccurred("BCE Failed4 ");
    return *this->loss;
}

Tensor<float>&  CrossEntropyLoss::backward(Tensor<int> &true_labels){
  // assert(this->D ==8);
  // std::cout << "Labels \n";
  // true_labels.debugTensor();
  // true_labels.viewTensor();
  // std::cout << "Predicted \n";
  // this->loss->debugTensor();
  // this->loss->viewTensor();
  cu_gradient<<<this->N, this->D>>>(this->exp_x->data_device, this->exp_sum->data_device, true_labels.data_device
        , this->dx->data_device, N, D);

  cudaDeviceSynchronize();
  return *this->dx;
}
