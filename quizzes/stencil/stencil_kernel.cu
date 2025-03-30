#include "stencil_kernel.h"
#include "utils/cuda.hpp"
#include <cstddef>
#include <torch/torch.h>
#include <torch/types.h>

// Only consider 3D case
template <typename scalar_t>
__constant__ scalar_t C[C_KERNEL_SIZE * C_KERNEL_SIZE * C_KERNEL_SIZE];

template <typename scalar_t>
__global__ void stencil_naive_kernel(const scalar_t *__restrict__ data,
                                     scalar_t *output, size_t N) {
  const unsigned int i = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int k = blockDim.x * blockIdx.x + threadIdx.x;
  // Note that we only stencil the element in [1..N-1, 1..N-1].
  // For element out of range, simply copy.
  if (i >= 1 && i < (N - 1) && j >= 1 && j < (N - 1) && k >= 1 && k < (N - 1)) {
    output[i * N * N + j * N + k] =
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER] *
            data[i * N * N + j * N + k] +
        C<scalar_t>[(C_CENTER - 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER] *
            data[(i - 1) * N * N + j * N + k] +
        C<scalar_t>[(C_CENTER + 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER] *
            data[(i + 1) * N * N + j * N + k] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    (C_CENTER + 1) * C_KERNEL_SIZE + C_CENTER] *
            data[i * N * N + (j + 1) * N + k] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    (C_CENTER - 1) * C_KERNEL_SIZE + C_CENTER] *
            data[i * N * N + (j - 1) * N + k] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER - 1] *
            data[i * N * N + j * N + k - 1] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER + 1] *
            data[i * N * N + j * N + k + 1];
  }
}

template <typename scalar_t>
__global__ void
stencil_shared_mem_tiling_kernel(const scalar_t *__restrict__ data,
                                 scalar_t *output, size_t N) {
  const int i = blockDim.z * blockIdx.z + threadIdx.z - 1;
  const int j = blockDim.y * blockIdx.y + threadIdx.y - 1;
  const int k = blockDim.x * blockIdx.x + threadIdx.x - 1;
  __shared__ scalar_t tile[IN_TILE_DIM * IN_TILE_DIM * IN_TILE_DIM];
  // Load data to shared memory
  // Index exceed boudary should be ignored.
  if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
    tile[threadIdx.z * OUT_TILE_DIM * OUT_TILE_DIM +
         threadIdx.y * OUT_TILE_DIM + threadIdx.x] =
        data[i * N * N + j * N + k];
  }
  __syncthreads();
  if (i >= 1 && i < (N - 1) && j >= 1 && j < (N - 1) && k >= 1 && k < (N - 1) &&
      threadIdx.x >= 1 && threadIdx.x < (IN_TILE_DIM - 1) && threadIdx.y >= 1 &&
      threadIdx.y < (IN_TILE_DIM - 1) && threadIdx.z >= 1 &&
      threadIdx.z < (IN_TILE_DIM - 1)) {
    output[i * N * N + j * N + k] =
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER] *
            tile[threadIdx.z * IN_TILE_DIM * IN_TILE_DIM +
                 threadIdx.y * IN_TILE_DIM + threadIdx.x] +
        C<scalar_t>[(C_CENTER - 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER] *
            tile[(threadIdx.z - 1) * IN_TILE_DIM * IN_TILE_DIM +
                 threadIdx.y * IN_TILE_DIM + threadIdx.x] +
        C<scalar_t>[(C_CENTER + 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER] *
            tile[(threadIdx.z + 1) * IN_TILE_DIM * IN_TILE_DIM +
                 threadIdx.y * IN_TILE_DIM + threadIdx.x] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    (C_CENTER + 1) * C_KERNEL_SIZE + C_CENTER] *
            tile[threadIdx.z * IN_TILE_DIM * IN_TILE_DIM +
                 (threadIdx.y + 1) * IN_TILE_DIM + threadIdx.x] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    (C_CENTER - 1) * C_KERNEL_SIZE + C_CENTER] *
            tile[threadIdx.z * IN_TILE_DIM * IN_TILE_DIM +
                 (threadIdx.y - 1) * IN_TILE_DIM + threadIdx.x] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER - 1] *
            tile[threadIdx.z * IN_TILE_DIM * IN_TILE_DIM +
                 threadIdx.y * IN_TILE_DIM + threadIdx.x - 1] +
        C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                    C_CENTER * C_KERNEL_SIZE + C_CENTER + 1] *
            tile[threadIdx.z * IN_TILE_DIM * IN_TILE_DIM +
                 threadIdx.y * IN_TILE_DIM + threadIdx.x + 1];
  }
}

torch::Tensor stencil_naive_cuda(torch::Tensor &data,
                                 torch::Tensor &coefficient) {

  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks =
      (data.size(1) - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  dim3 nblocks(blocks, blocks, blocks);
  dim3 nthreads(THREAD_PER_BLOCK, THREAD_PER_BLOCK, THREAD_PER_BLOCK);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "stencil_naive_kernel", [&] {
    cudaMemcpyToSymbol(C<scalar_t>, coefficient.const_data_ptr<scalar_t>(),
                       C_KERNEL_SIZE * C_KERNEL_SIZE * C_KERNEL_SIZE *
                           sizeof(scalar_t),
                       0, cudaMemcpyDeviceToDevice);
    stencil_naive_kernel<<<nblocks, nthreads>>>(data.const_data_ptr<scalar_t>(),
                                                output.data_ptr<scalar_t>(),
                                                data.size(1));
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
}

torch::Tensor stencil_shared_mem_tiling_cuda(torch::Tensor &data,
                                             torch::Tensor &coefficient) {
  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks = (data.size(1) - 1 + OUT_TILE_DIM) / OUT_TILE_DIM;

  dim3 nblocks(blocks, blocks, blocks);
  dim3 nthreads(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "stencil_shared_mem_tiling_kernel", [&] {
        cudaMemcpyToSymbol(C<scalar_t>, coefficient.const_data_ptr<scalar_t>(),
                           C_KERNEL_SIZE * C_KERNEL_SIZE * C_KERNEL_SIZE *
                               sizeof(scalar_t),
                           0, cudaMemcpyDeviceToDevice);
        stencil_naive_kernel<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            data.size(1));
      });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
}
