#include "kernel.h"
#include <torch/torch.h>
#include <torch/types.h>

template <typename scalar_t>
__global__ void mat_transpose_naive(const scalar_t *__restrict__ idata,
                                    float *__restrict__ odata, size_t M,
                                    size_t N) {

  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < N && y < M) {
    odata[x * M + y] = idata[y * N + x];
  }
}

torch::Tensor matrix_transpose_cuda(torch::Tensor &data) {
  auto M = data.size(0);
  auto N = data.size(1);
  auto output =
      torch::zeros({N, M}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const dim3 nblocks((M + BLOCKDIM - 1) / BLOCKDIM,
                     (N + BLOCKDIM - 1) / BLOCKDIM);
  const dim3 nthreads(BLOCKDIM, BLOCKDIM);

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "mat_transpose_naive", [&] {
    mat_transpose_naive<<<nblocks, nthreads>>>(data.const_data_ptr<scalar_t>(),
                                               output.data_ptr<float>(), M, N);
  });
  cudaDeviceSynchronize();
  return output;
}
