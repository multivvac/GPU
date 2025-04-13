#include "lenet_kernel.h"
#include "utils/cuda.hpp"
#include <ATen/Dispatch.h>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <cstdint>
#include <torch/torch.h>
__global__ void conv_forward_kernel() {};

template <typename scalar_t>
/**
 * @brief im2col CUDA kernel for unrolling convolution input data into matrix
 * format.
 *
 * This kernel transforms a 3D input image (C x H x W) into a 2D matrix where
 * each column is a flattened receptive field. This allows convolution to be
 * performed as matrix multiplication (GEMM).
 *
 * @param C      Number of input channels
 * @param H      Height of input image
 * @param W      Width of input image
 * @param K      Size of convolution filter
 * @param in     Pointer to input image data (size: C x H x W)
 * @param out    Pointer to output matrix (size: (C x R x S) x (H_out x W_out))
 */
__global__ void im2col_kernel(size_t C, size_t H, size_t W, size_t K,
                              const scalar_t *__restrict__ in,
                              scalar_t *__restrict__ out) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t H_out = H - K + 1;
  size_t W_out = H - K + 1;
  size_t W_unroll = H_out * W_out;

  if (tid < C * W_unroll) {
    size_t w_unroll = tid % W_unroll;
    size_t c = tid / W_unroll;
    size_t h = w_unroll / W_out;
    size_t w = w_unroll % W_out;
    for (size_t p = 0; p < K; p++) {
      for (size_t q = 0; q < K; q++) {
        size_t h_unroll = c * K * K + p * K + q;
        out[h_unroll * W_unroll + w_unroll] =
            in[c * H * W + (h + p) * W + w + q];
      }
    }
  }
}


__global__ void linear_forward_kernel() {};
__global__ void conv_backward() {};
__global__ void linear_backward() {};

torch::Tensor im2col_cuda(torch::Tensor &input, size_t K) {
  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);

  int64_t H_out = H - K + 1;
  int64_t W_out = W - K + 1;
  int64_t H_unroll = C * K * K;
  int64_t W_unroll = H_out * W_out;

  auto output =
      torch::zeros({N, H_unroll, W_unroll},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA));

  auto L = C * H_out * W_out;

  auto blocks = (L + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  for (size_t i = 0; i < N; i++) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "im2col_kernel", [&] {
          im2col_kernel<<<blocks, THREADS_PER_BLOCK>>>(
              C, H, W, K, input.const_data_ptr<scalar_t>() + i * C * H * W,
              output.data_ptr<scalar_t>() + i * H_unroll * W_unroll);
        });
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
};
