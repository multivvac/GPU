#ifndef STENCIL_KERNEL_H
#define STENCIL_KERNEL_H

#include <torch/torch.h>
#define ORDER 1
#define C_KERNEL_SIZE (2 * ORDER + 1)
#define C_CENTER C_KERNEL_SIZE / 2
#define IN_TILE_DIM 10
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * ORDER)
#define IN_TILE_DIM2 32
#define OUT_TILE_DIM2 (IN_TILE_DIM2 - 2 * ORDER)
#define THREAD_PER_BLOCK 8
torch::Tensor stencil_naive_cuda(torch::Tensor &data,
                                 torch::Tensor &coefficient);

torch::Tensor stencil_shared_mem_tiling_cuda(torch::Tensor &data,
                                             torch::Tensor &coefficient);

torch::Tensor stencil_thread_coarsening_cuda(torch::Tensor &data,
                                             torch::Tensor &coefficient);

torch::Tensor stencil_register_tiling_cuda(torch::Tensor &data,
                                           torch::Tensor &coefficient);
#endif
