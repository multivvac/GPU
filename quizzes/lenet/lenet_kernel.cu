#include "lenet_kernel.h"
#include "utils/cuda.hpp"
#include <ATen/Dispatch.h>
#include <c10/core/DeviceType.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
__launch_bounds__(THREADS_PER_BLOCK) __global__
    void im2col_kernel(size_t C, size_t H, size_t W, size_t K,
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

template <typename scalar_t>
// try to optimized with shared memory, but failed, naive version better
__global__ void im2col_optimized_kernel(size_t C, size_t H, size_t W, size_t K,
                                        const scalar_t *__restrict__ in,
                                        scalar_t *__restrict__ out) {

  size_t W_out = W - K + 1;
  size_t H_out = H - K + 1;
  size_t W_unroll = H_out * W_out;
  size_t section = blockIdx.y * W * H + blockIdx.x * W;
  extern __shared__ unsigned char smem[];
  scalar_t *in_s = reinterpret_cast<scalar_t *>(smem);
  for (size_t i = threadIdx.x; i < K * W; i++) {
    in_s[i] = in[section + i];
  }
  __syncthreads();
  for (size_t tid = threadIdx.x; tid < W_out; tid += blockDim.x) {
    size_t w_unroll = tid + W_out * blockIdx.x;
    for (size_t p = 0; p < K; p++) {
      for (size_t q = 0; q < K; q++) {
        size_t h_unroll = blockIdx.y * K * K + p * K + q;
        out[h_unroll * W_unroll + w_unroll] = in_s[tid + W * p + q];
      }
    }
  }
}

template <typename scalar_t>
__global__ void im2col_2d_optimized_kernel(size_t C, size_t H, size_t W,
                                           size_t K,
                                           const scalar_t *__restrict__ data_im,
                                           scalar_t *__restrict__ data_col) {
  size_t H_out = H - K + 1;
  size_t W_out = W - K + 1;

  size_t channel = blockIdx.z;

  extern __shared__ unsigned char smem[];
  scalar_t *data_im_s = reinterpret_cast<scalar_t *>(smem);

  // shared memory includes halo elements
  size_t H_shared = TILE_SIZE + K - 1;
  size_t W_shared = TILE_SIZE + K - 1;

  size_t tile_x = blockIdx.x * TILE_SIZE;
  size_t tile_y = blockIdx.y * TILE_SIZE;

  for (size_t j = threadIdx.y; j < H_shared; j += blockDim.y) {
    for (size_t i = threadIdx.x; i < W_shared; i += blockDim.x) {
      size_t imrow = tile_y + j;
      size_t imcol = tile_x + i;

      if (imrow < H && imcol < W) {
        data_im_s[i + W_shared * j] =
            data_im[channel * H * W + imrow * W + imcol];
      } else {
        data_im_s[i + W_shared * j] = static_cast<scalar_t>(0);
      }
    }
  }

  __syncthreads();

  for (size_t idy = threadIdx.y; idy < TILE_SIZE; idy += blockDim.y) {
    for (size_t idx = threadIdx.x; idx < TILE_SIZE; idx += blockDim.x) {
      size_t tx = tile_x + idx;
      size_t ty = tile_y + idy;
      size_t col_base = (channel * K * K) * H_out * W_out + ty * W_out + tx;
      if (tx < W_out && ty < H_out) {
        for (size_t p = 0; p < K; p++) {
          for (size_t q = 0; q < K; q++) {
            data_col[col_base + (p * K + q) * H_out * W_out] =
                data_im_s[(idy + p) * W_shared + idx + q];
          }
        }
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

torch::Tensor im2col_optimized_cuda(torch::Tensor &input, size_t K) {
  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);

  int64_t H_out = H - K + 1;
  int64_t W_out = W - K + 1;
  int64_t H_unroll = C * K * K;
  int64_t W_unroll = H_out * W_out;
  auto blocks = dim3((W_out + TILE_SIZE - 1) / TILE_SIZE,
                     (H_out + TILE_SIZE - 1) / TILE_SIZE, C);
  auto threads = dim3(std::min(32, TILE_SIZE), std::min(32, TILE_SIZE));

  auto output =
      torch::zeros({N, H_unroll, W_unroll},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA));

  for (size_t i = 0; i < N; i++) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "im2col_2d_optimized_kernel", [&] {
          im2col_2d_optimized_kernel<<<
              blocks, threads,
              (TILE_SIZE + K - 1) * (TILE_SIZE + K - 1) * sizeof(scalar_t)>>>(
              C, H, W, K, input.const_data_ptr<scalar_t>() + i * C * H * W,
              output.data_ptr<scalar_t>() + i * H_unroll * W_unroll);
        });
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
};
