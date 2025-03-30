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

template <typename scalar_t>
__global__ void
stencil_thread_coarsening_kernel(const scalar_t *__restrict__ data,
                                 scalar_t *output, size_t N) {
  int iStart = blockIdx.z * OUT_TILE_DIM2;
  const int j = OUT_TILE_DIM2 * blockIdx.y + threadIdx.y - 1;
  const int k = OUT_TILE_DIM2 * blockIdx.x + threadIdx.x - 1;
  __shared__ scalar_t tile_pre[IN_TILE_DIM2 * IN_TILE_DIM2];
  __shared__ scalar_t tile_cur[IN_TILE_DIM2 * IN_TILE_DIM2];
  __shared__ scalar_t tile_nxt[IN_TILE_DIM2 * IN_TILE_DIM2];

  if (iStart >= 1 && iStart < (N - 1) && j >= 0 && j < N && k >= 0 && k < N) {
    tile_pre[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        data[(iStart - 1) * N * N + j * N + k];
  }
  if (iStart < (N - 1) && j >= 0 && j < N && k >= 0 && k < N) {
    tile_cur[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        data[iStart * N * N + j * N + k];
  }
  __syncthreads();

  for (int i = iStart;
       i < (iStart + OUT_TILE_DIM2 > N - 1 ? N - 1 : (iStart + OUT_TILE_DIM2));
       i++) {
    if (j >= 0 && j < N && k >= 0 && k < N) {
      tile_nxt[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
          data[(i + 1) * N * N + j * N + k];
    }
    __syncthreads();
    if (i >= 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
        threadIdx.x >= 1 && threadIdx.x < (IN_TILE_DIM2 - 1) &&
        threadIdx.y >= 1 && threadIdx.y < (IN_TILE_DIM2 - 1)) {
      output[i * N * N + j * N + k] =
          C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      C_CENTER * C_KERNEL_SIZE + C_CENTER] *
              tile_cur[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] +
          C<scalar_t>[(C_CENTER - 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      C_CENTER * C_KERNEL_SIZE + C_CENTER] *
              tile_pre[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] +
          C<scalar_t>[(C_CENTER + 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      C_CENTER * C_KERNEL_SIZE + C_CENTER] *
              tile_nxt[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] +
          C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      (C_CENTER + 1) * C_KERNEL_SIZE + C_CENTER] *
              tile_cur[(threadIdx.y + 1) * IN_TILE_DIM2 + threadIdx.x] +
          C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      (C_CENTER - 1) * C_KERNEL_SIZE + C_CENTER] *
              tile_cur[(threadIdx.y - 1) * IN_TILE_DIM2 + threadIdx.x] +
          C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      C_CENTER * C_KERNEL_SIZE + C_CENTER - 1] *
              tile_cur[threadIdx.y * IN_TILE_DIM2 + threadIdx.x - 1] +
          C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                      C_CENTER * C_KERNEL_SIZE + C_CENTER + 1] *
              tile_cur[threadIdx.y * IN_TILE_DIM2 + threadIdx.x + 1];
    }
    __syncthreads();
    tile_pre[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        tile_cur[threadIdx.y * IN_TILE_DIM2 + threadIdx.x];
    tile_cur[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        tile_nxt[threadIdx.y * IN_TILE_DIM2 + threadIdx.x];
  }
}

// Below is the same thread corasening kernel as PMPP book.
// Copy here for reference
template <typename scalar_t>
__global__ void stencil_kernel(const scalar_t *__restrict__ in, scalar_t *out,
                               size_t N) {
  int iStart = blockIdx.z * OUT_TILE_DIM2;
  int j = blockIdx.y * OUT_TILE_DIM2 + threadIdx.y - 1;
  int k = blockIdx.x * OUT_TILE_DIM2 + threadIdx.x - 1;

  __shared__ scalar_t inPrev_s[IN_TILE_DIM2 * IN_TILE_DIM2];
  __shared__ scalar_t inCurr_s[IN_TILE_DIM2 * IN_TILE_DIM2];
  __shared__ scalar_t inNext_s[IN_TILE_DIM2 * IN_TILE_DIM2];

  // Load z-1 plane
  if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
    inPrev_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        in[(iStart - 1) * N * N + j * N + k];
  }

  // Load z plane
  if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
    inCurr_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        in[iStart * N * N + j * N + k];
  }

  // Load z+1 plane (loop over z)
  for (int i = iStart; i < iStart + OUT_TILE_DIM2; ++i) {
    if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
      inNext_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
          in[(i + 1) * N * N + j * N + k];
    }

    __syncthreads();

    // Only compute if in bounds and not halo
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (threadIdx.y > 0 && threadIdx.y < IN_TILE_DIM2 - 1 &&
          threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM2 - 1) {
        out[i * N * N + j * N + k] =
            C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        C_CENTER * C_KERNEL_SIZE + C_CENTER] *
                inCurr_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] +
            C<scalar_t>[(C_CENTER - 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        C_CENTER * C_KERNEL_SIZE + C_CENTER] *
                inPrev_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] +
            C<scalar_t>[(C_CENTER + 1) * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        C_CENTER * C_KERNEL_SIZE + C_CENTER] *
                inNext_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] +
            C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        (C_CENTER + 1) * C_KERNEL_SIZE + C_CENTER] *
                inCurr_s[(threadIdx.y + 1) * IN_TILE_DIM2 + threadIdx.x] +
            C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        (C_CENTER - 1) * C_KERNEL_SIZE + C_CENTER] *
                inCurr_s[(threadIdx.y - 1) * IN_TILE_DIM2 + threadIdx.x] +
            C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        C_CENTER * C_KERNEL_SIZE + C_CENTER - 1] *
                inCurr_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x - 1] +
            C<scalar_t>[C_CENTER * C_KERNEL_SIZE * C_KERNEL_SIZE +
                        C_CENTER * C_KERNEL_SIZE + C_CENTER + 1] *
                inCurr_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x + 1];
      }
    }

    __syncthreads();

    // Rotate buffers
    inPrev_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        inCurr_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x];
    inCurr_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x] =
        inNext_s[threadIdx.y * IN_TILE_DIM2 + threadIdx.x];
  }
}

torch::Tensor stencil_thread_coarsening_cuda(torch::Tensor &data,
                                             torch::Tensor &coefficient) {
  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks = (data.size(1) - 1 + OUT_TILE_DIM2) / OUT_TILE_DIM2;

  dim3 nblocks(blocks, blocks, blocks);
  dim3 nthreads(IN_TILE_DIM2, IN_TILE_DIM2, 1);
  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "stencil_thread_coarsening_kernel", [&] {
        cudaMemcpyToSymbol(C<scalar_t>, coefficient.const_data_ptr<scalar_t>(),
                           C_KERNEL_SIZE * C_KERNEL_SIZE * C_KERNEL_SIZE *
                               sizeof(scalar_t),
                           0, cudaMemcpyDeviceToDevice);
        stencil_thread_coarsening_kernel<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            data.size(1));
      });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
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
