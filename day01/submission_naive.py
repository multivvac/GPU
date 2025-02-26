#!POPCORN leaderboard histogram

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

histogram_cuda_source = """

template <typename scalar_t>
__global__ void histogram_kernel(const scalar_t *__restrict__ data,
                                  int* histogram, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  

  if (idx < N) {
    scalar_t bin = data[idx];
    atomicAdd(&histogram[bin], 1);
  }
}

torch::Tensor bincount_cuda(torch::Tensor data) {
  int N = data.numel();
  auto histogram = torch::zeros({256}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  const int threads = 128;
  const int blocks = (N - 1 + threads) / threads;

  AT_DISPATCH_INTEGRAL_TYPES(
      data.scalar_type(), "histogram_kernel", ([&] {
        histogram_kernel<scalar_t><<<blocks, threads>>>(
            data.data_ptr<scalar_t>(), histogram.data_ptr<int>(), N);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return histogram;
}

"""
histogram_cpp_source = """
#include <torch/extension.h>

torch::Tensor bincount_cuda(torch::Tensor data);
"""
bincount_module = load_inline(
    name="bincount_cuda",
    cpp_sources=histogram_cpp_source,
    cuda_sources=histogram_cuda_source,
    functions=["bincount_cuda"],
    extra_cuda_cflags=["-arch=sm_80"],
    verbose=True,
)


def bincount(data):
    if not data.is_cuda:
        raise RuntimeError("data tensor must be on GPU")
    return bincount_module.bincount_cuda(data)


def custom_kernel(data: input_t) -> output_t:
    return bincount(data)
