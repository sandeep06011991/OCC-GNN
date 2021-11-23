#include "loss.hh"
#include "nn_exception.hh"
#include<assert.h>

// ((float *)in.data_device, true_labels.data_device,
//       (float *) exp_sum->data_device,  loss->data_device, N, D);
__global__ void cu_loss(float *x, int * y, float *exp, float *loss, int N, int noClasses){
  int n = blockIdx.x;
  // int c = threadIdx.x;

  __shared__ float max;
  int offset = n * noClasses + threadIdx.x;
  if(threadIdx.x==0){
    max = 0;
      for(int j=0;j<noClasses;j++){
        if(max < x[n * noClasses + j])max = x[n * noClasses + j];
      }
    }
  __syncthreads();
  assert(blockDim.x < 20);
  __shared__ float sum[20];
  sum[threadIdx.x] = expf(x[offset] - max);
  exp[offset] = sum[threadIdx.x];
  assert(exp[offset]>=0);
  __syncthreads();
  __shared__ float exp_sum;
  if(threadIdx.x==0){
    exp_sum = 0;
    for(int i=0;i<noClasses;i++){
      exp_sum += sum[i];
    }
      loss[n] = -x[n*noClasses + y[n]] + max  + logf(exp_sum);
    }

  }


  // cu_gradient<<<this->N, this->D>>>(this->exp_x->data_device, true_labels.data_device
  //       , this->dx->data_device, N, D);
__global__ void cu_gradient(float *exp, int *y, float *out,  int N, int D){
  int n = blockIdx.x;
  int d = threadIdx.x;
  int global_id = n * blockDim.x + d;
  __shared__ float sum;
  if(threadIdx.x == 0){
    sum = 0;
    for(int i=0;i<D;i++){
      sum += exp[n*D + i];
    }
  }
  __syncthreads();
  assert(sum != 0 );
  float t = exp[global_id]/sum;
  if(threadIdx.x == y[n]){
    t = t - 1;
  }
  assert(N != 0);
  out[global_id] = t/N;
}

CrossEntropyLoss::CrossEntropyLoss(int n, int d,int device_id){
  this->N = n;
  this->D = d;
  this->device_id = device_id;
  this->exp_x = new Tensor<float>(Shape(n,d),this->device_id);
  // this->exp_sum = new Tensor<float>(n,1);
  this->loss = new Tensor<float>(Shape(n,1),this->device_id);
  this->dx = new Tensor<float>(Shape(n,d),this->device_id);
}

// void CrossEntropyLoss::compute_exponent(Tensor<float> &in){
//   int dim1 = in.dim1;
//   int dim2 = in.dim2;
//   // int noBlocks = ((in.dim1 * in.dim2) + 256)/256;
//   // int noThreads = 256;
//   std::cout << dim1 << " " << dim2 <<"\n";
//   cu_exponent<<<dim1,dim2>>>((float *) in.data_device,
//   (float *)  this->exp_x->data_device, dim1 * dim2);
//   NNException::throwIfDeviceErrorsOccurred("compute exponent failed ");
// }

// void CrossEntropyLoss::compute_exponent_sum(){
//   cu_exponent_sum<<<this->N,this->D>>>((float *) exp_x->data_device,
//   (float *)  exp_sum->data_device, this->N, this-> D);
//   NNException::throwIfDeviceErrorsOccurred("compute exponent sum failed ");
// }


void CrossEntropyLoss::compute_loss(Tensor<float> &in,
                Tensor<int> &true_labels){
  assert(in.s.dim1 == N);
  assert(in.s.dim2 == D);
  // float *x, int * y, float *exp, float *loss, int N, int noClasses
  cu_loss<<<in.s.dim1, in.s.dim2 >>>((float *)in.data_device, true_labels.data_device,
            (float *)this->exp_x->data_device,(float *) loss->data_device, N, D);
  NNException::throwIfDeviceErrorsOccurred("compute exponent loss failed ");
}

// For classification
// in Shape = N,C
// N num examples, C num examples.
// returns Tensor of shape N,1
Tensor<float>& CrossEntropyLoss::forward(Tensor<float> &in,Tensor<int> &true_labels){
    // in.debugTensor();
    this->N = in.s.dim1;
    this->D = in.s.dim2;
    int dim1 = in.s.dim1;
    int dim2 = in.s.dim2;
    // in.debugTensor();
    // in.viewTensor();
    // true_labels.debugTensor();
    // true_labels.viewTensor();
    if(this->exp_x != nullptr){
      this->exp_x->clearTensor();
      // this->exp_sum->clearTensor();
      this->loss->clearTensor();
      this->dx->clearTensor();
      delete this->exp_x;
      // delete this->exp_sum;
      delete this->loss;
      delete this->dx;
      NNException::throwIfDeviceErrorsOccurred("compute deletion failed ");
    }
    this->exp_x = new Tensor<float>(Shape(dim1,dim2),this->device_id);
    // this->exp_sum = new Tensor<float>(dim1,1);
    this->loss = new Tensor<float>(Shape(dim1,1),this->device_id);
    this->dx = new Tensor<float>(Shape(dim1,dim2),this->device_id);

    // compute_exponent(in);
    // NNException::throwIfDeviceErrorsOccurred("BCE Failed1 ");
    // exp_x->debugTensor();
    // exp_x->viewTensor();
    // compute_exponent_sum();
    // NNException::throwIfDeviceErrorsOccurred("BCE Failed2 ");
    // exp_sum->debugTensor();
    // exp_sum->viewTensor();
    cudaSetDevice(this->device_id);
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
  cudaSetDevice(this->device_id);
  cu_gradient<<<this->N, this->D>>>(this->exp_x->data_device, true_labels.data_device
        , this->dx->data_device, N, D);

  cudaDeviceSynchronize();
  return *this->dx;
}
