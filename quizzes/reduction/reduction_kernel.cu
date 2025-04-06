#include "reduction_kernel.h"
#include "utils/cuda.hpp"
#include <cstddef>
#include <torch/torch.h>
#include <torch/types.h>

template <typename scalar_t>
__global__ void reduction_naive_kernel(scalar_t *__restrict__ in,
                                       scalar_t *__restrict__ S, size_t N) {
  size_t section = 2 * blockIdx.x * blockDim.x;
  size_t tid = 2 * threadIdx.x;
  // note here the bound should be less or equal blockdim.x, because blockdim.x
  // is half of the size of 1 block
  for (size_t stride = 1; stride <= blockDim.x; stride *= 2) {
    size_t idx = tid * stride;
    if (idx < 2 * blockDim.x) {
      in[section + idx] += in[section + idx + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long int *>(&S[0]),
              static_cast<unsigned long long int>(in[section]));
  }
}

template <typename scalar_t>
__global__ void reduction_convergent_kernel(scalar_t *__restrict__ in,
                                            scalar_t *__restrict__ P,
                                            size_t N) {
  size_t section = 2 * blockIdx.x * blockDim.x;
  size_t tid = threadIdx.x;
  for (size_t stride = blockDim.x; stride >= 1; stride /= 2) {
    if (tid < stride) {
      in[section + tid] += in[section + tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    P[blockIdx.x] = in[section];
  }
}

template <typename scalar_t>
__global__ void reduction_shared_mem_kernel(scalar_t *__restrict__ in,
                                            scalar_t *__restrict__ P,
                                            size_t N) {
  size_t section = 2 * blockIdx.x * blockDim.x;
  size_t tid = threadIdx.x;
  __shared__ scalar_t in_s[MAX_THREAD_BLOCK];
  in_s[tid] = in[section + tid] + in[section + tid + blockDim.x];
  for (size_t stride = blockDim.x / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      in_s[tid] += in_s[tid + stride];
    }
  }
  if (tid == 0) {
    P[blockIdx.x] = in_s[0];
  }
}

torch::Tensor reduction_naive_cuda(torch::Tensor &data) {
  size_t N = data.numel();

  const size_t blocks = (N - 1 + 2 * THREAD_PER_BLOCK) / (2 * THREAD_PER_BLOCK);
  auto sum = torch::zeros(1, torch::dtype(torch::kInt64).device(torch::kCUDA));
  dim3 nblocks(blocks, 1, 1);
  dim3 nthreads(THREAD_PER_BLOCK, 1, 1);
  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "reduction_naive_kernel", [&] {
    reduction_naive_kernel<<<nblocks, nthreads>>>(data.data_ptr<scalar_t>(),
                                                  sum.data_ptr<scalar_t>(), N);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return sum;
}

torch::Tensor reduction_convergent_cuda(torch::Tensor &data) {
  size_t N = data.numel();
  // if we couldn't get result within 1 block.
  bool blockOverflow = (N / 2) > MAX_THREAD_BLOCK;

  const size_t threads = blockOverflow ? MAX_THREAD_BLOCK : (N / 2);
  const size_t blocks = (N - 1 + 2 * threads) / (2 * threads);
  auto partial =
      torch::zeros(blocks, torch::dtype(torch::kInt64).device(torch::kCUDA));
  dim3 nblocks(blocks, 1, 1);
  dim3 nthreads(threads, 1, 1);
  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "reduction_convergent_kernel", [&] {
    reduction_convergent_kernel<<<nblocks, nthreads>>>(
        data.data_ptr<scalar_t>(), partial.data_ptr<scalar_t>(), N);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return blockOverflow ? reduction_convergent_cuda(partial) : partial;
}

torch::Tensor reduction_shared_mem_cuda(torch::Tensor &data) {
  size_t N = data.numel();
  // if we couldn't get result within 1 block.
  bool blockOverflow = (N / 2) > MAX_THREAD_BLOCK;

  const size_t threads = blockOverflow ? MAX_THREAD_BLOCK : (N / 2);
  const size_t blocks = (N - 1 + 2 * threads) / (2 * threads);
  auto partial =
      torch::zeros(blocks, torch::dtype(torch::kInt64).device(torch::kCUDA));
  dim3 nblocks(blocks, 1, 1);
  dim3 nthreads(threads, 1, 1);
  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "reduction_shared_mem_kernel", [&] {
    reduction_shared_mem_kernel<<<nblocks, nthreads>>>(
        data.data_ptr<scalar_t>(), partial.data_ptr<scalar_t>(), N);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return blockOverflow ? reduction_convergent_cuda(partial) : partial;
}
