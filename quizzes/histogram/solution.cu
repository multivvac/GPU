#include "solution.h"
#include <cstdint>
#include <thread>
#include <torch/torch.h>
__global__ void histogram_kernel(const uint8_t *__restrict__ data,
                                 unsigned int *__restrict__ histo, int N) {

  __shared__ unsigned int localbins[NUM_BINS];

  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    localbins[i] = 0;
  }

  __syncthreads();

  if (idx < N) {
    atomicAdd(&(localbins[data[idx]]), 1);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histo[i], localbins[i]);
  }
}

torch::Tensor histogram_cuda(torch::Tensor &data) {

  int N = data.numel();
  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  const int blocks = (N - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  histogram_kernel<<<blocks, THREAD_PER_BLOCK>>>(
      data.const_data_ptr<uint8_t>(), histogram.data_ptr<unsigned int>(), N);
  return histogram;
}
