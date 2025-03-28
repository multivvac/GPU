#include "convolution_kernel.h"
#include "utils/cuda.hpp"
#include <cstddef>
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
                                         size_t height) {
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
  __shared__ scalar_t data_s[IN_TILE_DIM * IN_TILE_DIM];
  if (col < 0 || col >= width || row < 0 || row >= height) {
    data_s[threadIdx.y * IN_TILE_DIM + threadIdx.x] =
        static_cast<scalar_t>(0.0);
  } else {
    data_s[threadIdx.y * IN_TILE_DIM + threadIdx.x] = data[row * width + col];
  }
  scalar_t pValue = static_cast<scalar_t>(0.0);

  __syncthreads();

  size_t tileCol = threadIdx.x;
  size_t tileRow = threadIdx.y;

  col += FILTER_RADIUS;
  row += FILTER_RADIUS;
  if (row >= 0 && row < height && col >= 0 && col < width &&
      tileCol < OUT_TILE_DIM && tileRow < OUT_TILE_DIM) {
    for (size_t fRow = 0; fRow < FILTER_KERNEL_SIZE; fRow++) {
      for (size_t fCol = 0; fCol < FILTER_KERNEL_SIZE; fCol++) {
        pValue += F<scalar_t>[fRow * FILTER_KERNEL_SIZE + fCol] *
                  data_s[(tileRow + fRow) * IN_TILE_DIM + tileCol + fCol];
      }
    }
    output[row * width + col] = pValue;
  }
}

template <typename scalar_t>
__global__ void convolution_cached_2D_tiled_constant_mem_kernel(
    const scalar_t *__restrict__ data, float *output, size_t width,
    size_t height) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  __shared__ scalar_t data_s[TILE_DIM * TILE_DIM];

  // load data from global memory to L1 cache
  if (col < width && row < height) {
    data_s[threadIdx.x + threadIdx.y * TILE_DIM] = data[col + row * width];
  } else {
    data_s[threadIdx.x + threadIdx.y * TILE_DIM] = static_cast<scalar_t>(0.0);
  }
  __syncthreads();

  scalar_t pValue = static_cast<scalar_t>(0.0);
  if (col < width && row < height) {
    for (int fRow = 0; fRow < FILTER_KERNEL_SIZE; fRow++) {
      for (int fCol = 0; fCol < FILTER_KERNEL_SIZE; fCol++) {
        // Case when we are able to load from shared memory
        // Here we must write int sx/sy explicitly. If not then the code will be
        // buggy.
        int sx = threadIdx.x - FILTER_RADIUS + fCol;
        int sy = threadIdx.y - FILTER_RADIUS + fRow;
        if ((sx >= 0) && (sx < TILE_DIM) && (sy >= 0) && (sy < TILE_DIM)) {
          pValue += F<scalar_t>[fRow * FILTER_KERNEL_SIZE + fCol] *
                    data_s[sy * TILE_DIM + sx];
        } else {
          // edge case where we need to load from global memory(this
          // optimization aims to gain performance from L2 cache)
          if (((col - FILTER_RADIUS + fCol) >= 0) &&
              ((col - FILTER_RADIUS + fCol) < width) &&
              ((row - FILTER_RADIUS + fRow) < height) &&
              ((row - FILTER_RADIUS + fRow) >= 0)) {
            pValue += F<scalar_t>[fRow * FILTER_KERNEL_SIZE + fCol] *
                      data[(row - FILTER_RADIUS + fRow) * width + col -
                           FILTER_RADIUS + fCol];
          }
        }
      }
    }
    output[row * width + col] = pValue;
  }
}

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
  const size_t blocks = (data.size(1) - 1 + OUT_TILE_DIM) / OUT_TILE_DIM;

  dim3 nblocks(blocks, blocks, 1);
  dim3 nthreads(IN_TILE_DIM, IN_TILE_DIM, 1);

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
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
}

torch::Tensor convolution_cached_2D_tiled_constant_mem_cuda(
    torch::Tensor &data, torch::Tensor &filter_weight, int radius) {

  auto output = torch::zeros(
      data.sizes(), torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const size_t blocks = (data.size(1) - 1 + TILE_DIM) / TILE_DIM;

  dim3 nblocks(blocks, blocks, 1);
  dim3 nthreads(TILE_DIM, TILE_DIM, 1);

  AT_DISPATCH_ALL_TYPES(
      data.scalar_type(), "convolution_cached_2D_tiled_constant_mem_kernel",
      [&] {
        cudaMemcpyToSymbol(
            F<scalar_t>, filter_weight.const_data_ptr<scalar_t>(),
            FILTER_KERNEL_SIZE * FILTER_KERNEL_SIZE * sizeof(scalar_t), 0,
            cudaMemcpyDeviceToDevice);
        convolution_cached_2D_tiled_constant_mem_kernel<<<nblocks, nthreads>>>(
            data.const_data_ptr<scalar_t>(), output.data_ptr<float>(),
            data.size(1), data.size(2));
      });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return output;
}
