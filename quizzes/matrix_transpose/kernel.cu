#include "kernel.h"
#include <cstddef>
#include <torch/torch.h>
#include <torch/types.h>

template <typename scalar_t>
__global__ void mat_transpose_naive(const scalar_t *__restrict__ idata,
                                    float *__restrict__ odata, size_t M,
                                    size_t N) {

  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < N && y < M) {
    odata[x * M + y] = idata[y * N + x];
  }
}

template <typename scalar_t>
__global__ void mat_transpose_coalesced(const scalar_t *__restrict__ idata,
                                        float *__restrict__ odata, size_t M,
                                        size_t N) {

  __shared__ scalar_t tile[BLOCKDIM * BLOCKDIM];
  size_t x = BLOCKDIM * blockIdx.x + threadIdx.x;
  size_t y = BLOCKDIM * blockIdx.y + threadIdx.y;

  tile[threadIdx.y * BLOCKDIM + threadIdx.x] =
      (x < N && y < M) ? idata[y * N + x] : 0.0;
  __syncthreads();

  x = BLOCKDIM * blockIdx.y + threadIdx.x;
  y = BLOCKDIM * blockIdx.x + threadIdx.y;
  if (x < N && y < M) {
    odata[N * y + x] = tile[threadIdx.x * BLOCKDIM + threadIdx.y];
  }
}

template <typename scalar_t>
__global__ void
mat_transpose_coalesced_coarse_bank_conflict(const scalar_t *__restrict__ idata,
                                             float *__restrict__ odata,
                                             size_t M, size_t N) {

  __shared__ scalar_t tile[BLOCKDIM * BLOCKDIM];
  size_t x = BLOCKDIM * blockIdx.x + threadIdx.x;
  size_t y = BLOCKDIM * blockIdx.y + threadIdx.y;

  for (size_t i = 0; i < BLOCKDIM; i += blockDim.y) {
    tile[(threadIdx.y + i) * BLOCKDIM + threadIdx.x] = idata[(y + i) * N + x];
  }
  __syncthreads();

  x = BLOCKDIM * blockIdx.y + threadIdx.x;
  y = BLOCKDIM * blockIdx.x + threadIdx.y;
  for (size_t i = 0; i < BLOCKDIM; i += blockDim.y) {
    odata[N * (y + i) + x] = tile[threadIdx.x * BLOCKDIM + (threadIdx.y + i)];
  }
}

template <typename scalar_t>
__global__ void
mat_transpose_coalesced_coarse(const scalar_t *__restrict__ idata,
                               float *__restrict__ odata, size_t M, size_t N) {

  __shared__ scalar_t tile[BLOCKDIM * (BLOCKDIM + 1)];
  size_t x = BLOCKDIM * blockIdx.x + threadIdx.x;
  size_t y = BLOCKDIM * blockIdx.y + threadIdx.y;

  for (size_t i = 0; i < BLOCKDIM; i += blockDim.y) {
    tile[(threadIdx.y + i) * (BLOCKDIM + 1) + threadIdx.x] =
        idata[(y + i) * N + x];
  }
  __syncthreads();

  x = BLOCKDIM * blockIdx.y + threadIdx.x;
  y = BLOCKDIM * blockIdx.x + threadIdx.y;
  for (size_t i = 0; i < BLOCKDIM; i += blockDim.y) {
    odata[N * (y + i) + x] =
        tile[threadIdx.x * (BLOCKDIM + 1) + (threadIdx.y + i)];
  }
}
torch::Tensor matrix_transpose_cuda(torch::Tensor &data) {
  auto M = data.size(0);
  auto N = data.size(1);
  auto output =
      torch::zeros({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const dim3 nblocks((N + BLOCKDIM - 1) / BLOCKDIM,
                     (M + BLOCKDIM - 1) / BLOCKDIM);
  const dim3 nthreads(BLOCKDIM, BLOCKDIM);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "mat_transpose_naive", [&] {
    mat_transpose_naive<<<nblocks, nthreads>>>(data.const_data_ptr<scalar_t>(),
                                               output.data_ptr<float>(), M, N);
  });
  cudaDeviceSynchronize();
  return output;
}
torch::Tensor matrix_transpose_coalesced_cuda(torch::Tensor &data) {
  auto M = data.size(0);
  auto N = data.size(1);
  auto output =
      torch::zeros({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const dim3 nblocks((N + BLOCKDIM - 1) / BLOCKDIM,
                     (M + BLOCKDIM - 1) / BLOCKDIM);
  const dim3 nthreads(BLOCKDIM, BLOCKDIM);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "mat_transpose_coalesced", [&] {
    mat_transpose_coalesced<<<nblocks, nthreads>>>(
        data.const_data_ptr<scalar_t>(), output.data_ptr<float>(), M, N);
  });
  cudaDeviceSynchronize();
  return output;
}
torch::Tensor
matrix_transpose_coalesced_coarse_bank_conflict_cuda(torch::Tensor &data) {
  auto M = data.size(0);
  auto N = data.size(1);
  auto output =
      torch::zeros({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const dim3 nblocks((N + BLOCKDIM - 1) / BLOCKDIM,
                     (M + BLOCKDIM - 1) / BLOCKDIM);
  const dim3 nthreads(BLOCKDIM, BLOCKDIM / COARSE);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "mat_transpose_coalesced_coarse_bank_conflict", [&] {
        mat_transpose_coalesced_coarse_bank_conflict<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<float>(), M, N);
      });
  cudaDeviceSynchronize();
  return output;
}
torch::Tensor matrix_transpose_coalesced_coarse_cuda(torch::Tensor &data) {
  auto M = data.size(0);
  auto N = data.size(1);
  auto output =
      torch::zeros({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const dim3 nblocks((N + BLOCKDIM - 1) / BLOCKDIM,
                     (M + BLOCKDIM - 1) / BLOCKDIM);
  const dim3 nthreads(BLOCKDIM, BLOCKDIM / COARSE);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "mat_transpose_coalesced_coarse", [&] {
        mat_transpose_coalesced_coarse<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<float>(), M, N);
      });
  cudaDeviceSynchronize();
  return output;
}
