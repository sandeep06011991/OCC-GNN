#pragma once
#include "util/tensor.hh"

void mat_mul_a_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);

void mat_mul_a_b_t(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);

void mat_mul_a_t_b(Tensor<float>& A, bool transA, Tensor<float>& B, bool transB,
                          Tensor<float>& c);
