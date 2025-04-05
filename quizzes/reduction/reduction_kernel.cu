#include "reduction_kernel.h"
#include "utils/cuda.hpp"
#include <cstddef>
#include <torch/torch.h>
#include <torch/types.h>

template <typename scalar_t>
__global__ void reduction_naive_kernel(scalar_t *__restrict__ in,
                                       scalar_t *__restrict__ P, size_t N) {
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
    P[blockIdx.x] = in[section];
  }
}

torch::Tensor reduction_naive_cuda(torch::Tensor &data) {
  size_t N = data.numel();
  // if we couldn't get result within 1 block.
  bool blockOverflow = (N / 2) > MAX_THREAD_BLOCK;

  const size_t threads = blockOverflow ? MAX_THREAD_BLOCK : (N / 2);
  const size_t blocks = (N - 1 + 2 * threads) / (2 * threads);
  auto partial =
      torch::zeros(blocks, torch::dtype(torch::kInt64).device(torch::kCUDA));
  dim3 nblocks(blocks, 1, 1);
  dim3 nthreads(threads, 1, 1);
  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "reduction_naive_kernel", [&] {
    reduction_naive_kernel<<<nblocks, nthreads>>>(
        data.data_ptr<scalar_t>(), partial.data_ptr<scalar_t>(), N);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return blockOverflow ? reduction_naive_cuda(partial) : partial;
}
