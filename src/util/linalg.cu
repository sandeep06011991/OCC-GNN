#include "cublas_v2.h"

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

// ,mat mul in row major format
void mat_mul_a_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& C){
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t success= cublasSgemm( handle, CUBLAS_OP_N , CUBLAS_OP_N ,C.dim2, C.dim1
       , B.dim1 , \
       &alpha,   B.data_device, B.dim2, A.data_device, A.dim2 ,
        &beta , C.data_device, C.dim2);
    if ( success != CUBLAS_STATUS_SUCCESS){
                std::cout << "\33[31mErrorjjj: " << cublasGetErrorString(success) << "\33[0m\n";

        }
      cublasDestroy(handle);
}

// ,mat mul in row major format
void mat_mul_a_t_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& C){
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t success= cublasSgemm( handle,  CUBLAS_OP_N , CUBLAS_OP_T  ,C.dim2, C.dim1
       , B.dim1  , \
       &alpha,   B.data_device, B.dim2, A.data_device, A.dim2 ,
        &beta , C.data_device, C.dim2);
    if ( success != CUBLAS_STATUS_SUCCESS){
                std::cout << "\33[31mError: " << success << "\33[0m\n";

        }
    cublasDestroy(handle);
}

// ,mat mul in row major format
void mat_mul_a_b_t(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& C){
    float alpha = 1;
    float beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t success= cublasSgemm( handle, CUBLAS_OP_T , CUBLAS_OP_N ,C.dim2, C.dim1
       , B.dim2 , \
       &alpha,   B.data_device, B.dim2, A.data_device, A.dim2 ,
        &beta , C.data_device, C.dim2);
    if ( success != CUBLAS_STATUS_SUCCESS){
                std::cout << "\33[31mError: " << success << "\33[0m\n";

        }
    cublasDestroy(handle);
}
