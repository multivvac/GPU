#ifndef LENET_KERNEL_H
#define LENET_KERNEL_H

#include <torch/torch.h>
#define THREADS_PER_BLOCK 1024
#define TILE_SIZE 32
torch::Tensor im2col_cuda(torch::Tensor &input, size_t K);
torch::Tensor im2col_optimized_cuda(torch::Tensor &input, size_t K);
torch::Tensor conv2d_im2col_cuda(torch::Tensor &input, size_t K); 

#endif
