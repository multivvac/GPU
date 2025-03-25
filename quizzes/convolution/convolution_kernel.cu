#include "convolution_kernel.h"
#include <torch/torch.h>
#include <torch/types.h>

template <typename scalar_t>
__global__ void convolution_naive_kernel(const scalar_t *__restrict__ data,
                                         const scalar_t *__restrict__ filter,
                                         float *output, size_t r, size_t width,
                                         size_t height) {
  const size_t oCol = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t oRow = blockDim.x * blockIdx.x + threadIdx.x;

  const size_t kSize = 2 * r + 1;

  float Pvalue = 0.0f;

  for (size_t fCol = 0; fCol < kSize; fCol++) {
    for (size_t fRow = 0; fRow < kSize; fRow++) {
      int iRow = oRow - r + fRow;
      int iCol = oCol - r + fCol;
      if (iCol >= 0 && iCol < width && iRow < height && iRow >= 0) {
        Pvalue += data[iCol + width * iRow] * filter[fRow * kSize + fCol];
      }
    }
  }
  if (oRow < height && oCol < width) {
    output[oRow * width + oCol] = Pvalue;
  }
}

template <typename scalar_t>
__constant__ scalar_t F[FILTER_KERNEL_SIZE * FILTER_KERNEL_SIZE];

template <typename scalar_t>
__global__ void
convolution_constant_mem_kernel(const scalar_t *__restrict__ data,
                                float *output, size_t width, size_t height) {
  const size_t oCol = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t oRow = blockDim.x * blockIdx.x + threadIdx.x;

  float Pvalue = 0.0f;

  for (size_t fCol = 0; fCol < FILTER_KERNEL_SIZE; fCol++) {
    for (size_t fRow = 0; fRow < FILTER_KERNEL_SIZE; fRow++) {
      int iRow = oRow - FILTER_RADIUS + fRow;
      int iCol = oCol - FILTER_RADIUS + fCol;
      if (iCol >= 0 && iCol < width && iRow < height && iRow >= 0) {
        Pvalue += data[iCol + width * iRow] *
                  F<scalar_t>[fRow * FILTER_KERNEL_SIZE + fCol];
      }
    }
  }
  if (oRow < height && oCol < width) {
    output[oRow * width + oCol] = Pvalue;
  }
}

template <typename scalar_t>
__global__ void
convolution_2D_tiled_constant_mem_kernel(const scalar_t *__restrict__ data,
                                         float *output, size_t width,
                                         size_t height) {}

torch::Tensor convolution_naive_cuda(torch::Tensor &data,
                                     torch::Tensor &filter_weight, int radius) {

  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks =
      (data.size(1) - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  dim3 nblocks(blocks, blocks, 1);
  dim3 nthreads(THREAD_PER_BLOCK, THREAD_PER_BLOCK, 1);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "convolution_naive_kernel", [&] {
    convolution_naive_kernel<<<nblocks, nthreads>>>(
        data.const_data_ptr<scalar_t>(),
        filter_weight.const_data_ptr<scalar_t>(), output.data_ptr<float>(),
        radius, data.size(1), data.size(2));
  });
  cudaDeviceSynchronize();
  return output;
}

torch::Tensor convolution_constant_mem_cuda(torch::Tensor &data,
                                            torch::Tensor &filter_weight,
                                            int radius) {

  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks =
      (data.size(1) - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  dim3 nblocks(blocks, blocks, 1);
  dim3 nthreads(THREAD_PER_BLOCK, THREAD_PER_BLOCK, 1);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "convolution_constant_mem_kernel", [&] {
        cudaMemcpyToSymbol(
            F<scalar_t>, filter_weight.const_data_ptr<scalar_t>(),
            FILTER_KERNEL_SIZE * FILTER_KERNEL_SIZE * sizeof(scalar_t), 0,
            cudaMemcpyDeviceToDevice);
        convolution_constant_mem_kernel<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<float>(),
            data.size(1), data.size(2));
      });
  cudaDeviceSynchronize();
  return output;
}

torch::Tensor convolution_2D_tiled_constant_mem_cuda(
    torch::Tensor &data, torch::Tensor &filter_weight, int radius) {

  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks =
      (data.size(1) - 1 + THREAD_PER_BLOCK) / THREAD_PER_BLOCK;

  dim3 nblocks(blocks, blocks, 1);
  dim3 nthreads(THREAD_PER_BLOCK, THREAD_PER_BLOCK, 1);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "convolution_2D_tiled_constant_mem_kernel", [&] {
        cudaMemcpyToSymbol(
            F<scalar_t>, filter_weight.const_data_ptr<scalar_t>(),
            FILTER_KERNEL_SIZE * FILTER_KERNEL_SIZE * sizeof(scalar_t), 0,
            cudaMemcpyDeviceToDevice);
        convolution_2D_tiled_constant_mem_kernel<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<float>(),
            data.size(1), data.size(2));
      });
  cudaDeviceSynchronize();
  return output;
}
