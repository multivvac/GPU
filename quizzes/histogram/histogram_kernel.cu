#include "histogram_kernel.h"
#include <torch/torch.h>
#include <torch/types.h>

#define COARSE 32

template <typename scalar_t>
__global__ void histogram_kernel(const scalar_t *__restrict__ data,
                                 unsigned int *__restrict__ histo,
                                 unsigned long long N) {

  __shared__ unsigned int localbins[NUM_BINS];

  unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    localbins[i] = 0;
  }

  __syncthreads();

  if (idx < N) {
    unsigned int bin = static_cast<unsigned int>(data[idx]);
    atomicAdd_block(&(localbins[bin]), 1);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histo[i], localbins[i]);
  }
}

template <typename scalar_t>
__global__ void histogram_coarse_kernel(const scalar_t *__restrict__ data,
                                        unsigned int *__restrict__ histo,
                                        unsigned long long N) {

  __shared__ unsigned int localbins[NUM_BINS];

  unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    localbins[i] = 0;
  }

  __syncthreads();

  for (unsigned long long i = idx; i < N; i += blockDim.x * gridDim.x) {
    unsigned int bin = static_cast<unsigned int>(data[i]);
    atomicAdd_block(&(localbins[bin]), 1);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histo[i], localbins[i]);
  }
}

template <typename scalar_t>
__global__ void
histogram_coarse_contiguous_kernel(const scalar_t *__restrict__ data,
                                   unsigned int *__restrict__ histo,
                                   unsigned long long N) {

  __shared__ unsigned int localbins[NUM_BINS];

  unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    localbins[i] = 0;
  }

  __syncthreads();

  for (unsigned long long i = idx * COARSE; i < min((idx + 1) * COARSE, N);
       i++) {
    unsigned int bin = static_cast<unsigned int>(data[i]);
    atomicAdd_block(&(localbins[bin]), 1);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd_block(&histo[i], localbins[i]);
  }
}
template <typename scalar_t>
__global__ void histogram_vectorized_kernel(const scalar_t *__restrict__ data,
                                            unsigned int *__restrict__ histo,
                                            unsigned long long N) {

  __shared__ unsigned int localbins[NUM_BINS];

  unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    localbins[i] = 0;
  }

  __syncthreads();

  if (idx * 4 < N) {
    uchar4 v = reinterpret_cast<const uchar4 *>(data)[idx];
    atomicAdd_block(&(localbins[v.x]), 1);
    atomicAdd_block(&(localbins[v.y]), 1);
    atomicAdd_block(&(localbins[v.z]), 1);
    atomicAdd_block(&(localbins[v.w]), 1);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histo[i], localbins[i]);
  }
}

template <typename scalar_t>
__global__ void histogram_aggregation_kernel(const scalar_t *__restrict__ data,
                                             unsigned int *__restrict__ histo,
                                             unsigned long long N) {

  __shared__ unsigned int localbins[NUM_BINS];

  unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    localbins[i] = 0;
  }

  unsigned long long aggregation = 0;
  int pre_bin = -1;

  __syncthreads();

  for (unsigned long long i = idx; i < N; i += blockDim.x * gridDim.x) {
    unsigned int bin = data[i];
    if (bin == pre_bin) {
      aggregation += 1;
    } else {
      if (aggregation > 0) {
        atomicAdd_block(&(localbins[pre_bin]), aggregation);
      }
      aggregation = 1;
      pre_bin = bin;
    }
  }

  if (aggregation > 0) {
    atomicAdd_block(&(localbins[pre_bin]), aggregation);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    atomicAdd(&histo[i], localbins[i]);
  }
}

torch::Tensor histogram_cuda(torch::Tensor &data) {
  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  const unsigned long long N = data.numel();
  const unsigned long long blocks =
      (N - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "histogram_kernel", [&] {
    histogram_kernel<<<blocks, THREAD_PER_BLOCK>>>(
        data.const_data_ptr<scalar_t>(), histogram.data_ptr<unsigned int>(), N);
  });
  cudaDeviceSynchronize();
  return histogram;
}

torch::Tensor histogram_coarse_cuda(torch::Tensor &data) {
  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  const unsigned long long N = data.numel();
  const unsigned long long blocks =
      (N - 1 + THREAD_PER_BLOCK * COARSE) / (THREAD_PER_BLOCK * COARSE);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "histogram_coarse_kernel", [&] {
    histogram_coarse_kernel<<<blocks, THREAD_PER_BLOCK>>>(
        data.const_data_ptr<scalar_t>(), histogram.data_ptr<unsigned int>(), N);
  });
  cudaDeviceSynchronize();
  return histogram;
}

torch::Tensor histogram_coarse_contiguous_cuda(torch::Tensor &data) {
  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  const unsigned long long N = data.numel();
  const unsigned long long blocks =
      (N - 1 + THREAD_PER_BLOCK * COARSE) / (THREAD_PER_BLOCK * COARSE);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "histogram_coarse_contiguous_kernel", [&] {
        histogram_coarse_contiguous_kernel<<<blocks, THREAD_PER_BLOCK>>>(
            data.const_data_ptr<scalar_t>(), histogram.data_ptr<unsigned int>(),
            N);
      });
  cudaDeviceSynchronize();
  return histogram;
}

torch::Tensor histogram_vec_cuda(torch::Tensor &data) {

  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  const unsigned long long N = data.numel();
  const unsigned long long blocks =
      (N - 1 + THREAD_PER_BLOCK * 4) / (THREAD_PER_BLOCK * 4);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "histogram_vec_cuda", [&] {
    histogram_vectorized_kernel<<<blocks, THREAD_PER_BLOCK>>>(
        data.const_data_ptr<scalar_t>(), histogram.data_ptr<unsigned int>(), N);
  });
  cudaDeviceSynchronize();
  return histogram;
}

torch::Tensor histogram_aggregation_cuda(torch::Tensor &data) {
  auto histogram = torch::zeros(
      {NUM_BINS}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
  const unsigned long long N = data.numel();
  const unsigned long long blocks =
      (N - 1 + THREAD_PER_BLOCK * 4) / (THREAD_PER_BLOCK * 4);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "histogram_aggregation_kernel", [&] {
        histogram_aggregation_kernel<<<blocks, THREAD_PER_BLOCK>>>(
            data.const_data_ptr<scalar_t>(), histogram.data_ptr<unsigned int>(),
            N);
      });
  cudaDeviceSynchronize();
  return histogram;
}
