#include "solution.h"
#include <cstdint>
#include <torch/torch.h>
__global__ void histogram_kernel(const uint8_t *__restrict__ data,
                                 uint8_t *__restrict__ histo, int N) {}

torch::Tensor histogram_cuda(torch::Tensor &data) {

  int N = data.numel();
  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  const int blocks = (N - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  histogram_kernel<<<blocks, THREAD_PER_BLOCK>>>(
      data.const_data_ptr<uint8_t>(), histogram.data_ptr<uint8_t>(), N);
  return histogram;
}
