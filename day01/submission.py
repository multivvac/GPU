#!POPCORN leaderboard histogram

from torch.utils.cpp_extension import load_inline
from day01.task import input_t, output_t

histogram_cuda_source = """
#define CFACTOR 32
#define NUM_BINS 256
#define THREADS_PER_BLOCK 256

template <typename scalar_t>
__global__ void histogram_kernel(
    const scalar_t* __restrict__ data,
    int* histogram,
    int N)
{
    __shared__ int histo_s[NUM_BINS];

    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start = tid * CFACTOR;
    int end   = start + CFACTOR;
    if (end > N) {
        end = N;
    }

    for (int i = start; i < end; ++i) {
        int bin = static_cast<int>(data[i]);
        atomicAdd(&(histo_s[bin]), 1);
    }

    __syncthreads();

    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int val = histo_s[bin];
        if (val > 0) {
            atomicAdd(&(histogram[bin]), val);
        }
    }
}

torch::Tensor bincount_cuda(torch::Tensor data) {
  int N = data.numel();
  auto histogram = torch::zeros({NUM_BINS}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  const int blocks = (N - 1 + (THREADS_PER_BLOCK * CFACTOR)) / (THREADS_PER_BLOCK * CFACTOR);

  AT_DISPATCH_INTEGRAL_TYPES(
      data.scalar_type(), "histogram_kernel", ([&] {
        histogram_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
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
